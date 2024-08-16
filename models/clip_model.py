import torch.nn as nn
from transformers import BertModel

import models.encoders

class CodexCLIP(nn.Module):
    def __init__(self, marker_groups,
                 hf_model = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                 codex_dim = 384,
                 text_dim = 768,
                 projection_dim = 512):
        super(CodexCLIP, self).__init__()
        
        self.codex_encoder = models.encoders.CodexEncoder(marker_groups=marker_groups)
        self.text_encoder = BertModel.from_pretrained(hf_model)
        
        #TODO Finetuning parameters (should encoder params be frozen?)
        
        # Assuming codex_encoder and text_encoder produce different output dimensions
        self.codex_projection = nn.Linear(codex_dim, projection_dim)
        self.text_projection = nn.Linear(text_dim, projection_dim) 

    def forward(self, data):
        image_features = self.codex_projection(self.codex_encoder(data['codex']))
        text_features = self.text_projection(self.text_encoder(data['text']).last_hidden_state[:, 0, :])  # Use the CLS token output
        
        return {'codex': image_features, 
                'text': text_features}