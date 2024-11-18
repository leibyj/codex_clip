import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations

class PairwiseCLIPLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super(PairwiseCLIPLoss, self).__init__()
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


class OneVersusAllLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super(OneVersusAllLoss, self).__init__()
        self.temperature = temperature

    def forward(self, emb_dict: dict) -> torch.Tensor:
        """
        Args:
            emb_dict (dict): A dictionary where keys are modality names and values are 
                             the corresponding embeddings (Tensor) of shape (batch_size, embedding_dim).
                             It must contain exactly three modalities.
        """
        modalities = list(emb_dict.keys())
        assert len(modalities) == 3, "Exactly three modalities are required."

        emb_list = [F.normalize(emb_dict[mod], p=2, dim=-1) for mod in modalities]
        emb_all = torch.cat(emb_list, dim=0)  # shape (3*batch_size, embedding_dim)

        batch_size = emb_list[0].size(0)
        device = emb_list[0].device
        indices = torch.arange(batch_size, device=device)

        total_loss = 0.0

        for m_idx, emb_m in enumerate(emb_list):
            # Compute logits
            logits = emb_m @ emb_all.t() / self.temperature  # shape (batch_size, 3*batch_size)
            log_probs = F.log_softmax(logits, dim=1)

            # Create positive mask
            positive_mask = torch.zeros(batch_size, 3*batch_size, device=device, dtype=torch.bool)
            other_modalities = [idx for idx in range(3) if idx != m_idx]
            for o_idx in other_modalities:
                positive_mask[indices, indices + o_idx * batch_size] = True

            # Compute loss
            num_positives = len(other_modalities)
            loss = - (log_probs * positive_mask.float()).sum(dim=1) / num_positives
            loss = loss.mean()
            total_loss += loss

        # Average over modalities
        final_loss = total_loss / 3.0

        return final_loss
