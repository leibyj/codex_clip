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
    # load data

    workdir = "/Users/jacobleiby/Desktop/postdoc/Enable"

    with open(join(workdir, "data/s314_mIF+he+text.pkl"), "rb") as f:
        demo_data = pickle.load(f)

    # marker_groups = {'tumor': ["C4d", "DAPI"], 'stroma': ['CD38', 'Tbet', "DAPI"]}

    tokenizer = BertTokenizer.from_pretrained(cfg.model.text_model)
    train_data = CodexTextDataset(demo_data, tokenizer=tokenizer, max_len=cfg.model.max_length, tumor = ["C4d", "DAPI"], stroma = ['CD38', 'Tbet', "DAPI"])

    train_loader = DataLoader(train_data, batch_size=cfg.dataset.batch_size, shuffle=cfg.dataset.shuffle)

    # prep model and optimizer
    model = CodexCLIP(
        marker_groups=cfg.model.marker_groups,
        hf_model=cfg.model.text_model,
        codex_dim=cfg.model.codex_dim,
        text_dim=cfg.model.text_dim,
        projection_dim=cfg.model.projection_dim
    )

    # 
    if cfg.training.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    elif cfg.training.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training.learning_rate)
    
    # loss function
    criterion = MMCLIPLoss(temperature=cfg.loss.temperature)


    # Training loop
    for epoch in range(cfg.training.epochs):
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            embeddings = model(batch)
            loss = criterion(embeddings)
            print(loss)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{cfg.training.epochs}, Loss: {loss.item()}')

if __name__ == "__main__":
    train()
