import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, ShardingStrategy
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

from models import CodexCLIP_CVIT, CodexCLIP_CNN, CodexCNNTransformerIBOT
from data import MultiCodexTextDatasetSubset, MultiCodexTextDatasetFull, padded_collate_fn
from utils import MMCLIPLoss
 
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

def z_score_normalize(img, clip_range=None):
    mean = img.mean(dim=(1, 2), keepdim=True)
    std = img.std(dim=(1, 2), keepdim=True)
    
    normalized_img = (img - mean) / (std + 1e-6) 
    
    if clip_range is not None:
        normalized_img = torch.clamp(normalized_img, clip_range[0], clip_range[1])

    return normalized_img

def min_max_normalize(img):
    min_val = torch.amin(img, dim=(1, 2), keepdim=True)
    max_val = torch.amax(img, dim=(1, 2), keepdim=True)
    
    normalized_img = (img - min_val) / (max_val - min_val + 1e-6)  

    return normalized_img

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

    # sample_ids = ['s314_c002_v001_r002_reg004', 's314_c004_v001_r001_reg001',
    #                 's255_c001_v001_r001_reg002', 's255_c001_v001_r001_reg022']

    sample_ids = ['s314_c002_v001_r002_reg004', 's314_c004_v001_r001_reg001']

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
        # Load weights from CodexCNNTransformerIBOT into CodexCNNTransformer
        codex_cnn_transformer = model.codex_encoder  
        ibot_weights = torch.load('/project/zhihuanglab/jleiby/codex_clip/pretrained_encoder_weights/codex_cnn_transformer_ibot_pretrained.pth', map_location=device)
        codex_cnn_transformer_state_dict = codex_cnn_transformer.state_dict()
        filtered_ibot_weights = {k: v for k, v in ibot_weights.items() if k in codex_cnn_transformer_state_dict}
        codex_cnn_transformer_state_dict.update(filtered_ibot_weights)
        codex_cnn_transformer.load_state_dict(codex_cnn_transformer_state_dict)

        train_data = MultiCodexTextDatasetFull(cfg.dataset.path, sample_ids, tokenizer=tokenizer, max_len=cfg.dataset.max_length, transform=min_max_normalize)

    model = model.to(device)

    if n_gpus > 1:
        train_sampler = DistributedSampler(train_data, num_replicas=n_gpus, rank=rank)
        train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, num_workers = 16, sampler=train_sampler, collate_fn=padded_collate_fn, persistent_workers=True)
    else:
        train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, shuffle=cfg.dataset.shuffle)

    if n_gpus > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)

    if cfg.training.optimizer == "adam":
        # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
        optimizer = torch.optim.AdamW([
            {'params': model.module.text_encoder.parameters(), 'lr': cfg.training.text_lr, 'weight_decay': cfg.training.weight_decay},  # Text encoder parameters
            {'params': model.module.codex_encoder.parameters(), 'lr': cfg.training.codex_lr, 'weight_decay': cfg.training.weight_decay},  # CODEX encoder parameters
            {'params': model.module.codex_projection.parameters(), 'lr': cfg.training.codex_lr, 'weight_decay': cfg.training.weight_decay},  # CODEX projection layer
            {'params': model.module.text_projection.parameters(), 'lr': cfg.training.text_lr, 'weight_decay': cfg.training.weight_decay},  # Text projection layer
        ])
    elif cfg.training.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training.learning_rate)
    
    # Loss function
    criterion = MMCLIPLoss(temperature=cfg.loss.temperature).to(device)

    # Get codex data onto device, Dataset dependent
    if isinstance(train_data, MultiCodexTextDatasetSubset):
        def process_batch(batch):
            batch['codex'] = [(img.to(device), {'channels': channels['channels'].to(device)}) for img, channels in batch['codex']]
            return batch
    else:
        def process_batch(batch):
            batch['codex'] = batch['codex'].to(device)
            return batch
    
    scaler = GradScaler()  

    # Training loop
    for epoch in range(cfg.training.epochs):
        total_loss = 0
        model.train()

        if n_gpus > 1:
            train_sampler.set_epoch(epoch)

        for batch in tqdm(train_loader) if rank == 0 else train_loader:
            batch = process_batch(batch)
            batch['text'] = batch['text'].to(device)
            batch['att_mask'] = batch['att_mask'].to(device)

            optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                embeddings = model(batch)
                if n_gpus > 1:
                    embeddings = gather_embeddings(embeddings, n_gpus)
                    print(embeddings['text'].shape)
                loss = criterion(embeddings)

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if rank == 0:
                wandb.log({"batch_loss": loss.item()/batch['codex'].shape[0]})
                print(f"batch_loss = {loss.item()/batch['codex'].shape[0]}")

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
