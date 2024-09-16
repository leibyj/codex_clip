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

from models import CodexCNNTransformerIBOT
from data import MultiCodexTextDatasetSubset, MultiCodexTextDatasetFull, padded_collate_fn
from utils import z_score_normalize, min_max_normalize

import warnings
warnings.filterwarnings("ignore")

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def log_hydra_config_to_wandb(cfg):
    config_dict = OmegaConf.to_container(cfg, resolve=True) 
    wandb.config.update(config_dict, allow_val_change=True)

def gather_embeddings(embeddings_dict, world_size):
    gathered_embeddings_dict = {}
    for modality, embeddings in embeddings_dict.items():
        gathered_embeddings = dist_fn.all_gather(embeddings)
        gathered_embeddings = torch.cat(gathered_embeddings, dim=0)
        gathered_embeddings_dict[modality] = gathered_embeddings
    return gathered_embeddings_dict

def train(rank, cfg: DictConfig):
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

    sample_ids = ['s314_c002_v001_r002_reg004', 's314_c004_v001_r001_reg001',
                  's255_c001_v001_r001_reg002', 's255_c001_v001_r001_reg022']

    # Initialize the model for iBOT pretraining
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
    
    # Set up the dataset
    train_data = MultiCodexTextDatasetFull(cfg.dataset.path, sample_ids, tokenizer=tokenizer, max_len=cfg.dataset.max_length, transform=min_max_normalize)

    model = model.to(device)

    if n_gpus > 1:
        train_sampler = DistributedSampler(train_data, num_replicas=n_gpus, rank=rank)
        train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, num_workers=16, sampler=train_sampler, collate_fn=padded_collate_fn, persistent_workers=True)
    else:
        train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, shuffle=cfg.dataset.shuffle)

    if n_gpus > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Training loop
    for epoch in range(cfg.training.epochs):
        total_loss = 0
        model.train()

        if n_gpus > 1:
            train_sampler.set_epoch(epoch)

        for batch in tqdm(train_loader) if rank == 0 else train_loader:
            optimizer.zero_grad()

            # Process the batch for CodexCNNTransformerIBOT
            batch['codex'] = batch['codex'].to(device)

            # Forward pass
            loss = model(batch['codex'], batch['channels'])

            # Backpropagation
            loss.backward()
            optimizer.step()
            if hasattr(model, 'module'):
                model.module.update_teacher()
            else:
                # Single GPU or no DDP
                model.update_teacher()

            total_loss += loss.item()

            if rank == 0:
                wandb.log({"batch_loss": loss.item()})
                print(f"batch_loss = {loss.item()}")

        avg_loss = total_loss / len(train_loader)
        if rank == 0:
            wandb.log({"epoch_loss": avg_loss})
            print(f'Epoch {epoch+1}/{cfg.training.epochs}, Avg Loss: {avg_loss}')

    if rank == 0:
        wandb.finish()

    if n_gpus > 1:
        cleanup_ddp()

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        torch.multiprocessing.spawn(train, args=(cfg,), nprocs=n_gpus, join=True)
    else:
        train(0, cfg)

if __name__ == "__main__":
    main()
