import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations

class MMCLIPLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super(MMCLIPLoss, self).__init__()
        self.temperature = temperature

    def forward(self, emb_dict: dict) -> torch.Tensor:
        """
        Args:
            embeddings_dict (dict): A dictionary where keys are modality names and values are 
                                    the corresponding embeddings (Tensor) of shape (batch_size, embedding_dim).
        """
        # All pairwise combos
        modality_pairs = list(combinations(emb_dict.keys(), 2))
        total_loss = 0.0
        num_pairs = len(modality_pairs)

        for (modality_a, modality_b) in modality_pairs:
            
            emb_a = emb_dict[modality_a]
            emb_b = emb_dict[modality_b]

            emb_a = F.normalize(emb_a, p=2, dim=-1)
            emb_b = F.normalize(emb_b, p=2, dim=-1)

            # CLIP is bi-directional
            logits_a_to_b = emb_a @ emb_b.t() / self.temperature
            logits_b_to_a = emb_b @ emb_a.t() / self.temperature

            # Labels are indices of the ground-truth matches
            batch_size = emb_a.size(0)
            labels = torch.arange(batch_size, dtype=torch.long, device=emb_a.device)

            # CLIP is bi-directional
            loss_a_to_b = F.cross_entropy(logits_a_to_b, labels)
            loss_b_to_a = F.cross_entropy(logits_b_to_a, labels)

            # Add the average of the two losses to the total loss
            total_loss += (loss_a_to_b + loss_b_to_a) / 2

        # Average again?
        # final_loss = total_loss / num_pairs
        return total_loss