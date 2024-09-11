import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, ShardingStrategy
import torch.distributed.nn.functional as dist_fn
from torch import distributed as dist
from tqdm import tqdm
import wandb
from transformers import BertTokenizer
import pickle
from os.path import join
import os
import torch.nn.functional as F

from models import CodexCLIP_CVIT, CodexCLIP_CNN
from data import MultiCodexTextDatasetSubset, MultiCodexTextDatasetFull
from utils import MMCLIPLoss

def setup_fsdp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_fsdp():
    dist.destroy_process_group()

def log_hydra_config_to_wandb(cfg):
    config_dict = OmegaConf.to_container(cfg, resolve=True)  # resolve=True ensures that interpolations are resolved
    
    wandb.config.update(config_dict, allow_val_change=True)

def gather_embeddings(embeddings_dict, world_size):
    """
    Args:
        embeddings_dict (dict): Dictionary of embeddings from the local GPU.
        world_size (int): Total number of GPUs involved in training.
    """
    gathered_embeddings_dict = {}
    # Loop through each modality in the dictionary
    for modality, embeddings in embeddings_dict.items():
        gathered_embeddings = dist_fn.all_gather(embeddings)
        gathered_embeddings = torch.cat(gathered_embeddings, dim=0)
        gathered_embeddings_dict[modality] = gathered_embeddings

    return gathered_embeddings_dict


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
        log_hydra_config_to_wandb(cfg)

    print(f'Running on: {device}')

    # Load data
    # data = cfg.dataset.path

    # with open(join(workdir, f"data/{cfg.dataset.file}"), "rb") as f:
    #     demo_data = pickle.load(f)
    sample_ids = ['s314_c002_v001_r002_reg004', 's314_c004_v001_r001_reg001',
                    's255_c001_v001_r001_reg002', 's255_c001_v001_r001_reg022']

    tokenizer = BertTokenizer.from_pretrained(cfg.model.text_model)
    # train_data = MultiCodexTextDataset(cfg.dataset.path, sample_ids, tokenizer=tokenizer, max_len=cfg.dataset.max_length, **cfg.dataset.channel_groups)

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
            shared_projection=cfg.model.shared_projection # for CLIP model init
        )

        train_data = MultiCodexTextDatasetFull(cfg.dataset.path, sample_ids, tokenizer=tokenizer, max_len=cfg.dataset.max_length)

    model = model.to(device)

    def padded_collate_fn(batch):

        # Find the maximum number of channels in the batch
        max_channels = max(item['codex'].shape[0] for item in batch)

        # Prepare lists for the padded codex images, texts, and attention masks
        padded_codex_imgs = []
        texts = []
        attention_masks = []
        channels = []

        for item in batch:
            codex_img = item['codex']
            c, h, w = codex_img.shape

            # Pad the codex image to match the max number of channels in the batch
            if c < max_channels:
                padding = (0, 0, 0, 0, 0, max_channels - c)  # Pad only on the channel dimension
                padded_img = F.pad(codex_img, padding, mode='constant', value=0)
            else:
                padded_img = codex_img

            padded_codex_imgs.append(padded_img)
            texts.append(item['text'])
            attention_masks.append(item['att_mask'])
            channels.append(item['channels'])

        # Stack tensors to create batch
        padded_codex_imgs = torch.stack(padded_codex_imgs)
        texts = torch.stack(texts)
        attention_masks = torch.stack(attention_masks)

        return {
            "codex": padded_codex_imgs,
            "text": texts,
            "att_mask": attention_masks,
            "channels": channels
        }

    if n_gpus > 1:
        train_sampler = DistributedSampler(train_data, num_replicas=n_gpus, rank=rank)
        train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, sampler=train_sampler, collate_fn=padded_collate_fn)
    else:
        train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, shuffle=cfg.dataset.shuffle)


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


# Batch processing function based on dataset type
    if isinstance(train_data, MultiCodexTextDatasetSubset):
        def process_batch(batch):
            # For Subset: Handle grouped images
            batch['codex'] = [(img.to(device), {'channels': channels['channels'].to(device)}) for img, channels in batch['codex']]
            return batch
    else:
        def process_batch(batch):
            # For Full: Handle full images and channels
            batch['codex'] = batch['codex'].to(device)
            # batch['channels'] = [['PanCK', 'CD31', 'Vimentin', 'aSMA'] for i in batch['channels']]
            return batch
    
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
            embeddings = model(batch)

            if n_gpus > 1:
                embeddings = gather_embeddings(embeddings, n_gpus)

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
