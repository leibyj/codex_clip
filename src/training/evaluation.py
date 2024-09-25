import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from models.encoders import CodexCNNTransformerIBOT
from data import MultiCodexDatasetFull, padded_collate_fn_codex_only
from utils import MinMaxNormalize
import os
import numpy as np
import random
from tqdm import tqdm  # Add tqdm for progress bar

def load_checkpoint(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    print(f"Checkpoint loaded from: {checkpoint_path}")

def plot_embeddings(embeddings, labels, method='tsne', checkpoint_name=None):
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Method should be either 'tsne' or 'umap'")

    reduced_embeddings = reducer.fit_transform(embeddings)
    plt.figure(figsize=(12, 10))

    unique_regions = list(set(labels))
    color_map = plt.cm.get_cmap('tab10')
    colors = {region: color_map(i/len(unique_regions)) for i, region in enumerate(unique_regions)}

    for region_id in unique_regions:
        indices = [i for i, label in enumerate(labels) if label == region_id]
        plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], 
                    label=region_id, color=colors[region_id], alpha=0.7)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'Embeddings Visualization using {method.upper()} ({len(unique_regions)} Regions)')
    plt.tight_layout()
    
    # Create a directory for saving plots if it doesn't exist
    save_dir = '/project/zhihuanglab/jleiby/codex_clip/embedding_plots'
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate a unique filename based on method and checkpoint name
    filename = f'embeddings_{method}_{os.path.basename(checkpoint_name)}.png'
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free up memory
    
    print(f"Embedding plot saved at: {save_path}")
    
def run_inference_and_plot(model, data_loader, device, method='tsne', checkpoint_name=None):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Running Inference"):  # Add tqdm progress bar
            codex = batch['codex'].to(device)
            channels = batch['channels']
            embeddings = model(codex, channels, return_embeddings=True)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(batch['region_id'])

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    plot_embeddings(all_embeddings, all_labels, method, checkpoint_name)

def evaluate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(cfg.model.text_model)

    sample_ids = pd.read_csv('/project/zhihuanglab/jleiby/codex_clip/sample_names.csv', header=None)[0].tolist()
    eval_sample_ids = sample_ids[20:30]  # Use the first 10 sample IDs for evaluation

    model = CodexCNNTransformerIBOT(
        vocab=cfg.model.vocab,
        embed_dim=cfg.model.codex_dim, 
        nhead=cfg.model.nhead, 
        num_layers=cfg.model.num_layers, 
        mask_ratio=0
    )

    eval_data = MultiCodexDatasetFull(cfg.dataset.path, eval_sample_ids, tokenizer, transform=None)
    eval_loader = DataLoader(eval_data, batch_size=cfg.dataset.batch_size, num_workers=24, shuffle=False, collate_fn=padded_collate_fn_codex_only)

    checkpoint_dir = '/project/zhihuanglab/jleiby/codex_clip/checkpoints'
    checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]

    plot_dir = '/project/zhihuanglab/jleiby/codex_clip/embedding_plots'
    os.makedirs(plot_dir, exist_ok=True)

    for checkpoint_path in checkpoint_files:
        load_checkpoint(model, checkpoint_path)
        model.to(device)
        run_inference_and_plot(model, eval_loader, device, method='tsne', checkpoint_name=checkpoint_path)

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="../configs", config_name="config")
    def main(cfg: DictConfig):
        evaluate(cfg)

    main()