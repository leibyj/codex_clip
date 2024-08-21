import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from transformers import BertTokenizer
import pickle
from os.path import join


from models import CodexCLIP
from data import CodexTextDataset
from utils import MMCLIPLoss

@hydra.main(config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
    print(f'Running on: {device}')

    # load data
    workdir = "/Users/jacobleiby/Documents/GitHub/codex_clip"

    with open(join(workdir, "data/s314_mIF+he+text.pkl"), "rb") as f:
        demo_data = pickle.load(f)

    tokenizer = BertTokenizer.from_pretrained(cfg.model.text_model)
    train_data = CodexTextDataset(demo_data, tokenizer=tokenizer, max_len=cfg.dataset.max_length, **cfg.dataset.channel_groups)

    train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, shuffle=cfg.dataset.shuffle)

    # prep model and optimizer
    model = CodexCLIP(
        marker_groups=cfg.model.marker_groups,
        hf_model=cfg.model.text_model,
        codex_dim=cfg.model.codex_dim,
        text_dim=cfg.model.text_dim,
        projection_dim=cfg.model.projection_dim,
        shared_projection=cfg.model.shared_projection,
        device=device
    ).to(device)

    # 
    if cfg.training.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    elif cfg.training.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training.learning_rate)
    
    # loss function
    criterion = MMCLIPLoss(temperature=cfg.loss.temperature).to(device)


    # Training loop
    for epoch in range(cfg.training.epochs):
        model.train()
        for batch in tqdm(train_loader):
            # batch = {key: val.to(device) for key, val in batch.items()}
            batch['codex'] = [(img.to(device), {'channels': channels['channels'].to(device)}) for img, channels in batch['codex']]
            batch['text'] = batch['text'].to(device)

            optimizer.zero_grad()
            embeddings = model(batch)
            loss = criterion(embeddings)
            print(loss)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{cfg.training.epochs}, Loss: {loss.item()}')

if __name__ == "__main__":
    train()
