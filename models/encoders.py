
import torch
import torch.nn as nn

class CodexEncoder(nn.Module):
    def __init__(self, marker_groups=0, hidden_dim=384):
        super(CodexEncoder, self).__init__()

        # initialize ChannelViTs for each marker group        
        self.channelvit = [self._load_channelvit_model() for i in range(marker_groups)]

        # project combined embeddings to lower dimension (combined_dim (3 * 384) -> MHA embed dim)
        self.projection = nn.Sequential(
            nn.Linear(marker_groups*384, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )

        # Self attention layer
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=0.1)
        self.attn_postprocess = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )

    def _load_channelvit_model(self):
        model = torch.hub.load('insitro/ChannelViT', 'cpjump_cellpaint_bf_channelvit_small_p8_with_hcs_supervised', pretrained=False)
        state_dict = torch.load("/Users/jacobleiby/Downloads/cpjump_cellpaint_bf_channelvit_small_p8_with_hcs_supervised.pth", map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        return model
    
    def forward(self, x):

        cls_tokens = []
        for i, dat in enumerate(x):
            cls_tokens.append(self.channelvit[i](dat[0], dat[1]))

        # Concatenate CLS embeddings along the feature dimension and project to CA embedding size
        combined_cls = torch.cat(cls_tokens, dim=1)  # Shape: (batch_size, maker_group * embedding_dim)
        combined_cls = self.projection(combined_cls)
        # Reshape for self attention (needs 3D tensor: (sequence_length, batch_size, embed_dim))
        combined_cls = combined_cls.unsqueeze(0)  # Shape: (1, batch_size, embedding_dim)

        # Perform self attention
        attn_output, _ = self.self_attention(combined_cls, combined_cls, combined_cls)
        attn_output = self.attn_postprocess(attn_output)

        return attn_output.squeeze(0)