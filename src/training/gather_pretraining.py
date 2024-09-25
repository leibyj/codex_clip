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
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from datetime import timedelta

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
    transforms.RandomResizedCrop(256),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    MinMaxNormalize(), 
])

def get_batch_lr_lambda(cfg, total_batches):
    def lr_lambda(batch):
        if batch < cfg.training.warmup_batches:
            return float(batch) / float(max(1, cfg.training.warmup_batches))
        return 1.0  # Keep the learning rate constant after the warmup phase
    return lr_lambda

def save_checkpoint(model, epoch, batch_idx, checkpoint_dir='/project/zhihuanglab/jleiby/codex_clip/checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pth')
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

        sample_ids = pd.read_csv('/project/zhihuanglab/jleiby/codex_clip/sample_names.csv', header=None)[0].tolist()
        train_sample_ids = sample_ids
        # eval_sample_ids = sample_ids[10:20]  # Use the first 10 sample IDs for evaluation

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
        
        train_data = MultiCodexDatasetFull(cfg.dataset.path, train_sample_ids, transform=data_transforms)
        # eval_data = MultiCodexDatasetFull(cfg.dataset.path, eval_sample_ids, tokenizer, transform=None)

        model = model.to(device)

        if n_gpus > 1:
            train_sampler = DistributedSampler(train_data, num_replicas=n_gpus, rank=rank)
            train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, num_workers=32, sampler=train_sampler, collate_fn=padded_collate_fn_codex_only, persistent_workers=True, pin_memory=True)
            # eval_loader = DataLoader(eval_data, batch_size=cfg.dataset.batch_size, num_workers=4, sampler=DistributedSampler(eval_data, num_replicas=n_gpus, rank=rank), collate_fn=padded_collate_fn_codex_only, persistent_workers=True, pin_memory=True)
        else:
            train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, shuffle=cfg.dataset.shuffle)
            # eval_loader = DataLoader(eval_data, batch_size=cfg.dataset.batch_size, shuffle=False)

        if n_gpus > 1:
            model = DDP(model, device_ids=[rank], output_device=rank)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
        
        if cfg.training.amp:
            scaler = GradScaler()  # For mixed precision training
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

        total_batches = len(train_loader) * cfg.training.epochs
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_batch_lr_lambda(cfg, total_batches))

        for epoch in range(cfg.training.epochs):
            total_loss = 0
            model.train()

            if n_gpus > 1:
                train_sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(tqdm(train_loader) if rank == 0 else train_loader):
                optimizer.zero_grad()

                batch['codex'] = batch['codex'].to(device)

                if cfg.training.amp:
                    with autocast():
                        loss, cls_loss, mim_loss = model(batch['codex'], batch['channels'])
                else:
                    loss, cls_loss, mim_loss = model(batch['codex'], batch['channels'])

                if cfg.training.amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.max_grad_norm)
                    optimizer.step()

                warmup_scheduler.step()

                if hasattr(model, 'module'):
                    model.module.update_teacher()
                else:
                    model.update_teacher()

                total_loss += loss.item()

                if batch_idx == 0 or batch_idx == (len(train_loader) // 2):
                    if rank == 0:
                        save_checkpoint(model, epoch, batch_idx)

                if rank == 0:
                    wandb.log({"batch_loss": loss.item(),
                               "CLS_loss": cls_loss.item(),
                               "MIM_loss": mim_loss.item()})
                    print(f"batch_loss = {loss.item()}, CLS_loss = {cls_loss.item()}, MIM_loss = {mim_loss.item()}")

            avg_loss = total_loss / len(train_loader)
            scheduler.step()

            if rank == 0:
                wandb.log({"epoch_loss": avg_loss})
                print(f'Epoch {epoch+1}/{cfg.training.epochs}, Avg Loss: {avg_loss}, LR: {scheduler.get_last_lr()[0]}')

        if rank == 0:
            save_dir = '/project/zhihuanglab/jleiby/codex_clip/pretrained_encoder_weights'  # Replace with your desired path
            save_path = os.path.join(save_dir, 'codex_cnn_transformer_ibot_pretrained.pth')
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"Model saved successfully at {save_path}.")
        if rank == 0:
            wandb.finish()

        if n_gpus > 1:
            cleanup_ddp()
    except Exception as e:
        print(f"Exception in rank {rank}: {e}")
        if n_gpus > 1:
            cleanup_ddp()
        raise e

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        torch.multiprocessing.spawn(train, args=(cfg,), nprocs=n_gpus, join=True)
    else:
        train(0, cfg)

if __name__ == "__main__":
    main()
