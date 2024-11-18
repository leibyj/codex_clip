import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed.nn.functional as dist_fn
from torch import distributed as dist
from tqdm import tqdm
import wandb
from transformers import BertTokenizer
import pickle
from os.path import join
import os
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from models import CodexCNNTransformerIBOT
from data import MultiCodexTextDatasetSubset, MultiCodexTextDatasetFull, padded_collate_fn, MultiCodexDatasetFull, padded_collate_fn_codex_only
from utils import z_score_normalize, min_max_normalize, MinMaxNormalize
from evaluation import *
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import csv

import warnings
warnings.filterwarnings("ignore")

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    timeout = timedelta(minutes=30) 
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def log_hydra_config_to_wandb(cfg):
    config_dict = OmegaConf.to_container(cfg, resolve=True) 
    wandb.config.update(config_dict, allow_val_change=True)

data_transforms = transforms.Compose([
    # transforms.RandomResizedCrop(256),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    MinMaxNormalize(), 
])

def get_batch_lr_lambda(cfg, total_batches):
    def lr_lambda(batch):
        if batch < cfg.training.warmup_batches:
            return float(batch) / float(max(1, cfg.training.warmup_batches))
        return 1.0  # Keep the learning rate constant after the warmup phase
    return lr_lambda

def save_checkpoint(model, epoch, batch_idx, run_dir):
    checkpoint_path = os.path.join(run_dir, f'ibot_checkpoint_epoch_{epoch}_batch_{batch_idx}.pth')
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), checkpoint_path)
    else:
        torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")

def train(rank, cfg: DictConfig):
    try:
        n_gpus = torch.cuda.device_count()
        
        if n_gpus > 1:
            setup_ddp(rank, n_gpus)
            device = torch.device("cuda", rank)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if rank == 0:
            wandb.init(
                project=cfg.wandb.project_name,
                name=cfg.wandb.run_name,
            )
            log_hydra_config_to_wandb(cfg)

        print(f'Running on: {device}')

        tokenizer = BertTokenizer.from_pretrained(cfg.model.text_model)

        sample_ids = pd.read_csv('/project/zhihuanglab/jleiby/codex_clip/sample_names_20241007.csv', header=None)[0].tolist()
        exclude_ids = ['s624_c001_v001_r001_reg001', 'ffx-50241', 'hxs-77087', 'UofChica-89_c001_v001_r001_reg001']  # Replace with the actual IDs you want to exclude
        train_sample_ids = [sample_id for sample_id in sample_ids if sample_id not in exclude_ids]
        # train_sample_ids = [sample_id for sample_id in sample_ids if sample_id.startswith("UPMC")]

        model = CodexCNNTransformerIBOT(
            vocab=cfg.model.vocab,
            embed_dim=cfg.model.codex_dim, 
            nhead=cfg.model.nhead, 
            num_layers=cfg.model.num_layers, 
            mask_ratio=cfg.model.mask_ratio,
            teacher_temperature=cfg.model.teacher_temperature,
            student_temperature=cfg.model.student_temperature,
            momentum=cfg.model.momentum
        )

        train_data = MultiCodexDatasetFull(cfg.dataset.path, train_sample_ids, transform=None)

        model = model.to(device)

        if n_gpus > 1:
            train_sampler = DistributedSampler(train_data, num_replicas=n_gpus, rank=rank, shuffle=cfg.dataset.shuffle, seed = 1001)
            train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, num_workers=24, sampler=train_sampler, collate_fn=padded_collate_fn_codex_only, persistent_workers=True, pin_memory=True)
        else:
            train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, num_workers=24, shuffle=True, collate_fn=padded_collate_fn_codex_only, persistent_workers=True, pin_memory=True)

        if n_gpus > 1:
            model = DDP(model, device_ids=[rank], output_device=rank)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
        
        if cfg.training.amp:
            scaler = GradScaler()  # For mixed precision training
        
        if cfg.evaluation.run_eval_during_training:
            eval_train_loader, eval_test_loader, num_labels = prep_eval_dataloaders(cfg.evaluation.data_path, cfg.evaluation.label_file_path, cfg.evaluation.batch_size)

        total_batches = len(train_loader) * cfg.training.epochs
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_batch_lr_lambda(cfg, total_batches))
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_batches - cfg.training.warmup_batches)

        if rank == 0:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M')
            run_dir = os.path.join('/project/zhihuanglab/jleiby/codex_clip/checkpoints', timestamp)
            os.makedirs(run_dir, exist_ok=True)

            # Save the full config as a text file
            config_path = os.path.join(run_dir, f'{cfg.wandb.run_name}_config.yaml')
            with open(config_path, 'w') as f:
                f.write(OmegaConf.to_yaml(cfg))
            print(f"Configuration saved at: {config_path}")

        accumulation_steps = 4  # Number of batches to accumulate gradients
        global_batch_idx = 0 
        for epoch in range(cfg.training.epochs):
            total_loss = 0
            model.train()

            if n_gpus > 1:
                train_sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(tqdm(train_loader) if rank == 0 else train_loader):
                if n_gpus == 1:
                    optimizer.zero_grad() if (batch_idx + 1) % accumulation_steps == 0 else None
                else:
                    optimizer.zero_grad()

                if batch_idx % (len(train_loader) // 10) == 0:
                    if rank == 0:
                        save_checkpoint(model, epoch, batch_idx, run_dir)

                batch['codex'] = batch['codex'].to(device)

                if cfg.training.amp:
                    with autocast():
                        loss, cls_loss, mim_loss = model(batch['codex'], batch['channels'])
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)  # Unscale the gradients
                    if cfg.training.gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_value)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss, cls_loss, mim_loss = model(batch['codex'], batch['channels'])
                    loss.backward()
                    if cfg.training.gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_value)
                    optimizer.step()

                if global_batch_idx < cfg.training.warmup_batches:
                    warmup_scheduler.step()
                else:
                    cosine_scheduler.step()

                if hasattr(model, 'module'):
                    model.module.update_teacher()
                else:
                    model.update_teacher()

                total_loss += loss.item()

                if rank == 0:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "CLS_loss": cls_loss.item(),
                        "MIM_loss": mim_loss.item(),
                        "learning_rate": optimizer.param_groups[0]['lr']  # Log the learning rate
                    })
                    print(f"batch_loss = {loss.item()}, CLS_loss = {cls_loss.item()}, MIM_loss = {mim_loss.item()}, LR = {optimizer.param_groups[0]['lr']}")

                if cfg.evaluation.run_eval_during_training:
                    if global_batch_idx % cfg.evaluation.steps == 0:
                        torch.distributed.barrier()
                        eval_metrics = run_linear_probe_torch(model, eval_train_loader, eval_test_loader, num_labels, input_dim=cfg.model.codex_dim, device=device)
                        if rank == 0:
                            write_eval_metrics(run_dir, epoch, global_batch_idx, eval_metrics)
                        torch.distributed.barrier()
                        model.train()
                
                global_batch_idx += 1

            avg_loss = total_loss / len(train_loader)
        
            if rank == 0:
                wandb.log({"epoch_loss": avg_loss})
                print(f'Epoch {epoch+1}/{cfg.training.epochs}, Avg Loss: {avg_loss}, LR: {optimizer.param_groups[0]["lr"]}')

        if rank == 0:
            save_dir = '/project/zhihuanglab/jleiby/codex_clip/pretrained_encoder_weights'  # Replace with your desired path
            save_checkpoint(model, cfg.training.epochs, 'final', save_dir)
            print(f"Final model saved successfully.")

        if n_gpus > 1:
            cleanup_ddp()
    except Exception as e:
        print(f"Exception in rank {rank}: {e}")
        if n_gpus > 1:
            cleanup_ddp()
        raise e

def write_eval_metrics(checkpoint_dir, epoch, batch_idx, eval_metrics):
    csv_file = os.path.join(checkpoint_dir, 'evaluation_metrics.csv')
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write the header if the file does not exist
            writer.writerow(['epoch', 'batch_idx', 'train_mse', 'test_mse'])
        
        # Write the metrics
        writer.writerow([epoch, batch_idx, eval_metrics['train_mse'], eval_metrics['test_mse']])

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        torch.multiprocessing.spawn(train, args=(cfg,), nprocs=n_gpus, join=True)
    else:
        train(0, cfg)

if __name__ == "__main__":
    main()
