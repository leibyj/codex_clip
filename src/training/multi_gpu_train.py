import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, ShardingStrategy
from torch import distributed as dist
from tqdm import tqdm
import wandb
from transformers import BertTokenizer
import pickle
from os.path import join
import os

from models import CodexCLIP
from data import CodexTextDataset, MultiCodexTextDataset
from utils import MMCLIPLoss

def setup_fsdp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_fsdp():
    dist.destroy_process_group()

def train(rank, cfg: DictConfig):
    n_gpus = torch.cuda.device_count()
    
    # Setup FSDP if more than one GPU is available
    if n_gpus > 1:
        setup_fsdp(rank, n_gpus)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        wandb.init(
            project=cfg.wandb.project_name,
            name=cfg.wandb.run_name,
        )

    print(f'Running on: {device}')

    # Load data
    # data = cfg.dataset.path

    # with open(join(workdir, f"data/{cfg.dataset.file}"), "rb") as f:
    #     demo_data = pickle.load(f)

    tokenizer = BertTokenizer.from_pretrained(cfg.model.text_model)
    train_data = MultiCodexTextDataset(cfg.dataset.path, ['s255_c001_v001_r001_reg002', 's255_c001_v001_r001_reg022'], tokenizer=tokenizer, max_len=cfg.dataset.max_length, **cfg.dataset.channel_groups)

    if n_gpus > 1:
        train_sampler = DistributedSampler(train_data, num_replicas=n_gpus, rank=rank)
        train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, shuffle=cfg.dataset.shuffle)

    # Prep model and optimizer
    model = CodexCLIP(
        marker_groups=cfg.model.marker_groups,
        hf_model=cfg.model.text_model,
        codex_dim=cfg.model.codex_dim,
        text_dim=cfg.model.text_dim,
        projection_dim=cfg.model.projection_dim,
        shared_projection=cfg.model.shared_projection,
        device=device,
        pt_path=cfg.model.pt_path
    ).to(device)

    # Wrap the model with FSDP instead of DDP
    if n_gpus > 1:
        model = FSDP(
            model,
            cpu_offload=CPUOffload(offload_params=True),  # Offload parameters to CPU if needed
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # Fully sharded mode
        )

    if cfg.training.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    elif cfg.training.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training.learning_rate)
    
    # Loss function
    criterion = MMCLIPLoss(temperature=cfg.loss.temperature).to(device)

    # Training loop
    for epoch in range(cfg.training.epochs):
        total_loss = 0
        model.train()
        
        if n_gpus > 1:
            train_sampler.set_epoch(epoch)

        for batch in tqdm(train_loader) if rank == 0 else train_loader:
            batch['codex'] = [(img.to(device), {'channels': channels['channels'].to(device)}) for img, channels in batch['codex']]
            batch['text'] = batch['text'].to(device)
            batch['att_mask'] = batch['att_mask'].to(device)

            optimizer.zero_grad()
            embeddings = model(batch)
            loss = criterion(embeddings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if rank == 0:
            print(f'Epoch {epoch+1}/{cfg.training.epochs}, Loss: {avg_loss}')
            wandb.log({"loss": avg_loss})

    if rank == 0:
        wandb.finish()

    if n_gpus > 1:
        cleanup_fsdp()

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        torch.multiprocessing.spawn(train, args=(cfg,), nprocs=n_gpus, join=True)
    else:
        train(0, cfg)

if __name__ == "__main__":
    main()
