import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed.nn.functional as dist_fn
from torch import distributed as dist
from torchvision import transforms
from tqdm import tqdm
import wandb
from transformers import BertTokenizer
import pickle
from os.path import join
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import random

from models import CodexCLIP_CVIT, CodexCLIP_CNN, CodexCNNTransformerIBOT, TrimodalEncoder, BimodalEncoder
from data import MultiCodexTextDatasetSubset, MultiCodexTextDatasetFull, padded_collate_fn, TrimodalDataset, padded_collate_fn_trimodal
from utils import PairwiseCLIPLoss, OneVersusAllLoss
from datetime import timedelta, datetime
import csv

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gather_embeddings(embeddings_dict, world_size):
    """
    Args:
        embeddings_dict (dict): Dictionary of embeddings from the local GPU.
        world_size (int): Total number of GPUs involved in training.
    """
    gathered_embeddings_dict = {}
    # Loop through each modality in the dictionary returned from model
    for modality, embeddings in embeddings_dict.items():
        gathered_embeddings = dist_fn.all_gather(embeddings)
        gathered_embeddings = torch.cat(gathered_embeddings, dim=0)
        gathered_embeddings_dict[modality] = gathered_embeddings

    return gathered_embeddings_dict

def get_batch_lr_lambda(cfg, total_batches):
    def lr_lambda(batch):
        if batch < cfg.training.warmup_batches:
            return float(batch) / float(max(1, cfg.training.warmup_batches))
        return 1.0  # Keep the learning rate constant after the warmup phase
    return lr_lambda

def save_checkpoint(model, epoch, batch_idx, run_dir):
    checkpoint_path = os.path.join(run_dir, f'clip_checkpoint_epoch_{epoch}_batch_{batch_idx}.pth')
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), checkpoint_path)
    else:
        torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")

# def z_score_normalize(img, clip_range=None):
#     mean = img.mean(dim=(1, 2), keepdim=True)
#     std = img.std(dim=(1, 2), keepdim=True)
    
#     normalized_img = (img - mean) / (std + 1e-6) 
    
#     if clip_range is not None:
#         normalized_img = torch.clamp(normalized_img, clip_range[0], clip_range[1])

#     return normalized_img

# def min_max_normalize(img):
#     min_val = torch.amin(img, dim=(1, 2), keepdim=True)
#     max_val = torch.amax(img, dim=(1, 2), keepdim=True)
    
#     normalized_img = (img - min_val) / (max_val - min_val + 1e-6)  

#     return normalized_img

def train(rank, cfg: DictConfig):
    n_gpus = torch.cuda.device_count()
    set_seed(seed=1001)
    
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

    # sample_ids = sample_ids = pd.read_csv('/project/zhihuanglab/jleiby/codex_clip/sample_ids_with_text.csv', header=None)[0].tolist()
    sample_ids = sample_ids = pd.read_csv('/project/zhihuanglab/jleiby/enable_data/UPMC_s4065_reindexed_full_region_ids.csv', header=None)[0].tolist()

    tokenizer = BertTokenizer.from_pretrained(cfg.model.text_model)

    # transform = min_max_normalize()

    # Prep model and dataloader
    if cfg.model.name == 'codex_text_cvit':
        model = CodexCLIP_CVIT(
            marker_groups=cfg.model.marker_groups, # for codex encoder model init
            hf_model=cfg.model.text_model, # for CLIP model init
            codex_dim=cfg.model.codex_dim, # for CLIP model init
            text_dim=cfg.model.text_dim, # for CLIP model init
            projection_dim=cfg.model.projection_dim, # for CLIP model init
            shared_projection=cfg.model.shared_projection, # for CLIP model init
            device=device, # for codex encoder model init
            pt_path=cfg.model.pt_path # for codex encoder model init
        )

        train_data = MultiCodexTextDatasetSubset(cfg.dataset.path, sample_ids, tokenizer=tokenizer, max_len=cfg.dataset.max_length, **cfg.dataset.channel_groups)
        if n_gpus > 1:
            train_sampler = DistributedSampler(train_data, num_replicas=n_gpus, rank=rank, shuffle=cfg.dataset.shuffle, seed = 1001)
            train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, num_workers = 16, sampler=train_sampler, collate_fn=padded_collate_fn, persistent_workers=True, pin_memory=True)
        else:
            train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, shuffle=cfg.dataset.shuffle)
        
    elif cfg.model.name == 'codex_text_cnn':
        model = CodexCLIP_CNN(
            vocab=cfg.model.vocab,
            embed_dim=cfg.model.codex_dim, # for codex encoder model init
            nhead=cfg.model.nhead, # for codex encoder model init
            num_layers=cfg.model.num_layers, # for codex encoder model init
            hf_model=cfg.model.text_model, # for CLIP model init
            codex_dim=cfg.model.codex_dim, # for CLIP model init
            text_dim=cfg.model.text_dim, # for CLIP model init
            projection_dim=cfg.model.projection_dim, # for CLIP model init
            shared_projection=cfg.model.shared_projection, # for CLIP model init
            freeze_bert_layers=True, # for CLIP model init
            tune_bert_layers=[10, 11] # for CLIP model init
        )
        
        if cfg.model.load_from_pt:
            codex_cnn_transformer = model.codex_encoder  
            ibot_weights = torch.load(cfg.model.pt_model_path, map_location=device)
            codex_cnn_transformer_state_dict = codex_cnn_transformer.state_dict()
            filtered_ibot_weights = {k: v for k, v in ibot_weights.items() if k in codex_cnn_transformer_state_dict}
            codex_cnn_transformer_state_dict.update(filtered_ibot_weights)
            codex_cnn_transformer.load_state_dict(codex_cnn_transformer_state_dict)

        train_data = MultiCodexTextDatasetFull(cfg.dataset.path, sample_ids, tokenizer=tokenizer, max_len=cfg.dataset.max_length, transform=None)
        if n_gpus > 1:
            train_sampler = DistributedSampler(train_data, num_replicas=n_gpus, rank=rank, shuffle=cfg.dataset.shuffle, seed = 1001)
            train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, num_workers = 16, sampler=train_sampler, collate_fn=padded_collate_fn, persistent_workers=True, pin_memory=True)
        else:
            train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, shuffle=cfg.dataset.shuffle)
        
    elif cfg.model.name == 'trimodal':
        model = TrimodalEncoder(
            hf_model=cfg.model.text_model, # for CLIP model init
            codex_dim=cfg.model.codex_dim, # for CLIP model init
            text_dim=cfg.model.text_dim, # for CLIP model init
            he_dim=cfg.model.he_dim, # for CLIP model init
            projection_dim=cfg.model.projection_dim, # for CLIP model init
            shared_projection=cfg.model.shared_projection, # for CLIP model init
            freeze_bert_layers=True, # for CLIP model init
            tune_bert_layers=[10, 11], # for CLIP model init
            vocab=cfg.model.vocab,
            nhead=cfg.model.nhead, # for codex encoder model init
            num_layers=cfg.model.num_layers, # for codex encoder model init
        )
        
        if cfg.model.load_from_pt:
            codex_cnn_transformer = model.codex_encoder  
            ibot_weights = torch.load(cfg.model.pt_model_path, map_location=device)
            codex_cnn_transformer_state_dict = codex_cnn_transformer.state_dict()
            filtered_ibot_weights = {k: v for k, v in ibot_weights.items() if k in codex_cnn_transformer_state_dict}
            codex_cnn_transformer_state_dict.update(filtered_ibot_weights)
            codex_cnn_transformer.load_state_dict(codex_cnn_transformer_state_dict)

        # HandE transforms
        he_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])
        train_data = TrimodalDataset(cfg.dataset.path, sample_ids, tokenizer=tokenizer, max_len=cfg.dataset.max_length, codex_transform=None, he_transform=he_transform)
        if n_gpus > 1:
            train_sampler = DistributedSampler(train_data, num_replicas=n_gpus, rank=rank, shuffle=cfg.dataset.shuffle, seed = 1001)
            train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, num_workers = 16, sampler=train_sampler, collate_fn=padded_collate_fn_trimodal, persistent_workers=True, pin_memory=True)
        else:
            train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, shuffle=cfg.dataset.shuffle)
        
    elif cfg.model.name == 'trimodal':
        model = BimodalEncoder(
            modalities=cfg.model.modalities,
            hf_model=cfg.model.text_model, # for CLIP model init
            codex_dim=cfg.model.codex_dim, # for CLIP model init
            text_dim=cfg.model.text_dim, # for CLIP model init
            he_dim=cfg.model.he_dim, # for CLIP model init
            projection_dim=cfg.model.projection_dim, # for CLIP model init
            shared_projection=cfg.model.shared_projection, # for CLIP model init
            freeze_bert_layers=True, # for CLIP model init
            tune_bert_layers=[10, 11], # for CLIP model init
            vocab=cfg.model.vocab,
            nhead=cfg.model.nhead, # for codex encoder model init
            num_layers=cfg.model.num_layers, # for codex encoder model init
        )
        
        # if cfg.model.load_from_pt:
        #     codex_cnn_transformer = model.codex_encoder  
        #     ibot_weights = torch.load(cfg.model.pt_model_path, map_location=device)
        #     codex_cnn_transformer_state_dict = codex_cnn_transformer.state_dict()
        #     filtered_ibot_weights = {k: v for k, v in ibot_weights.items() if k in codex_cnn_transformer_state_dict}
        #     codex_cnn_transformer_state_dict.update(filtered_ibot_weights)
        #     codex_cnn_transformer.load_state_dict(codex_cnn_transformer_state_dict)

        # HandE transforms
        he_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])
        train_data = TrimodalDataset(cfg.dataset.path, sample_ids, tokenizer=tokenizer, max_len=cfg.dataset.max_length, codex_transform=None, he_transform=he_transform)
        if n_gpus > 1:
            train_sampler = DistributedSampler(train_data, num_replicas=n_gpus, rank=rank, shuffle=cfg.dataset.shuffle, seed = 1001)
            train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, num_workers = 16, sampler=train_sampler, collate_fn=padded_collate_fn_trimodal, persistent_workers=True, pin_memory=True)
        else:
            train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, shuffle=cfg.dataset.shuffle)

    model = model.to(device)
    if n_gpus > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)

    if cfg.training.optimizer == "adam":
        # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
        optimizer = torch.optim.AdamW([
            {'params': model.module.text_encoder.parameters(), 'lr': cfg.training.text_lr, 'weight_decay': cfg.training.weight_decay},  # text encoder parameters
            {'params': model.module.text_projection.parameters(), 'lr': cfg.training.text_lr, 'weight_decay': cfg.training.weight_decay},  # text projection layer
            {'params': model.module.codex_encoder.parameters(), 'lr': cfg.training.codex_lr, 'weight_decay': cfg.training.weight_decay},  # CODEX encoder parameters
            {'params': model.module.codex_projection.parameters(), 'lr': cfg.training.codex_lr, 'weight_decay': cfg.training.weight_decay},  # CODEX projection layer
            {'params': model.module.he_encoder.parameters(), 'lr': cfg.training.he_lr, 'weight_decay': cfg.training.weight_decay},  # HE encoder parameters
            {'params': model.module.he_projection.parameters(), 'lr': cfg.training.he_lr, 'weight_decay': cfg.training.weight_decay}  # HE projection layer
        ])
    elif cfg.training.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training.learning_rate)
    
    total_batches = len(train_loader) * cfg.training.epochs
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_batch_lr_lambda(cfg, total_batches))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_batches - cfg.training.warmup_batches)

    criterion = PairwiseCLIPLoss(temperature=cfg.loss.temperature).to(device)
    # criterion = OneVersusAllLoss(temperature=cfg.loss.temperature).to(device)

    if isinstance(train_data, MultiCodexTextDatasetSubset):
        def process_batch(batch):
            batch['codex'] = [(img.to(device), {'channels': channels['channels'].to(device)}) for img, channels in batch['codex']]
            return batch
    else:
        def process_batch(batch):
            batch['codex'] = batch['codex'].to(device)
            batch['text'] = batch['text'].to(device)
            batch['att_mask'] = batch['att_mask'].to(device)
            batch['HandE'] = batch['HandE'].to(device)
            return batch
    
    scaler = GradScaler()  

    if rank == 0:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M')
        run_dir = os.path.join('/project/zhihuanglab/jleiby/codex_clip/checkpoints', f"{cfg.model.name}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        # Save the full config as a text file
        config_path = os.path.join(run_dir, f'{cfg.wandb.run_name}_config.yaml')
        with open(config_path, 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))
        print(f"Configuration saved at: {config_path}")

    global_batch_idx = 0 

    for epoch in range(cfg.training.epochs):
        total_loss = 0
        model.train()

        if n_gpus > 1:
            train_sampler.set_epoch(epoch)

        # for batch in tqdm(train_loader) if rank == 0 else train_loader:
        for batch_idx, batch in enumerate(tqdm(train_loader) if rank == 0 else train_loader):
            optimizer.zero_grad()

            if batch_idx % (len(train_loader) // 2) == 0:
                if rank == 0:
                    save_checkpoint(model, epoch, batch_idx, run_dir)

            batch = process_batch(batch)

            with autocast():
                embeddings = model(batch)
                if n_gpus > 1:
                    embeddings = gather_embeddings(embeddings, n_gpus)
                    # print(embeddings['text'].shape)
                loss = criterion(embeddings)

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.training.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_value)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if global_batch_idx < cfg.training.warmup_batches:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

            if rank == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                    })
                # print(f"batch_loss = {loss.item()}")

        avg_loss = total_loss / len(train_loader)
        if rank == 0:
            wandb.log({"epoch_loss": avg_loss})
            print(f'Epoch {epoch+1}/{cfg.training.epochs}, Avg Loss: {avg_loss}')

    if rank == 0:
        save_dir = '/project/zhihuanglab/jleiby/codex_clip/pretrained_encoder_weights'  # Replace with your desired path
        save_checkpoint(model, cfg.training.epochs, 'final', save_dir)
        print(f"Final model saved successfully.")
        wandb.finish()

    if n_gpus > 1:
        cleanup_ddp()

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    n_gpus = torch.cuda.device_count()
    set_seed(1001)

    if n_gpus > 1:
        torch.multiprocessing.spawn(train, args=(cfg,), nprocs=n_gpus, join=True)
    else:
        train(0, cfg)

if __name__ == "__main__":
    main()
