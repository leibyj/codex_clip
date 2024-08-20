import torch.nn as nn
from transformers import BertModel

import models.encoders

class CodexCLIP(nn.Module):
    def __init__(self, marker_groups,
                 hf_model = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                 codex_dim = 384,
                 text_dim = 768,
                 projection_dim = 512,
                 shared_projection = False):
        super(CodexCLIP, self).__init__()
        
        self.codex_encoder = models.encoders.CodexEncoder(marker_groups=marker_groups)
        self.text_encoder = BertModel.from_pretrained(hf_model)
        
        #TODO Finetuning parameters (where should encoder params be trained/frozen?)
        
        # if shared_projection, check if dims are equal, if not map them to equal... then share projection. Otherwise idependent projection layers
        if shared_projection:
            if codex_dim == text_dim:
                shared = nn.Linear(codex_dim, projection_dim)
                self.codex_projection = shared
                self.text_projection = shared
            else:
                # map CODEX to text dim, then shared projection
                shared = nn.Linear(text_dim, projection_dim)
                self.codex_projection = nn.Sequential(
                    nn.Linear(codex_dim, text_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.LayerNorm(text_dim),
                    shared
                )
                self.text_projection = shared

        else: 
            # independent projections
            self.codex_projection = nn.Linear(codex_dim, projection_dim)
            self.text_projection = nn.Linear(text_dim, projection_dim) 

    def forward(self, data):
        image_features = self.codex_projection(self.codex_encoder(data['codex']))
        text_features = self.text_projection(self.text_encoder(data['text']).last_hidden_state[:, 0, :])  # Use the CLS token output
        
        return {'codex': image_features, 
                'text': text_features}