import torch.nn as nn
from transformers import BertModel
import models.encoders

class BaseCodexCLIP(nn.Module):
    def __init__(self, 
                 hf_model="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                 codex_dim=384,
                 text_dim=768,
                 projection_dim=512,
                 shared_projection=False,
                 freeze_bert_layers=False,
                 tune_bert_layers=None):
        """
        Args:
            hf_model (str): The Hugging Face model identifier for the text encoder.
            codex_dim (int): The dimensionality of the Codex encoder output.
            text_dim (int): The dimensionality of the text encoder output.
            projection_dim (int): The dimensionality of the projection layers.
            shared_projection (bool): If True, a shared projection layer is used for both Codex and text features.
        """
        super(BaseCodexCLIP, self).__init__()
        
        # Initialize the text encoder
        self.text_encoder = BertModel.from_pretrained(hf_model)
       
        if freeze_bert_layers and tune_bert_layers is not None:
            self._freeze_bert_layers(tune_bert_layers)
        else:
            for param in self.text_encoder.parameters():
                param.requires_grad = True
        # Handle shared or independent projection layers
        if shared_projection:
            if codex_dim == text_dim:
                shared = nn.Linear(codex_dim, projection_dim)
                self.codex_projection = shared
                self.text_projection = shared
            else:
                # Map CODEX to text dim, then shared projection
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
            # Independent projections
            self.codex_projection = nn.Linear(codex_dim, projection_dim)
            self.text_projection = nn.Linear(text_dim, projection_dim)
        
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Applies Xavier initialization to the linear layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _freeze_bert_layers(self, tune_bert_layers):
        """
        Args:
            tune_bert_layers (list): List of BERT layers to keep trainable. All others will be frozen.
        """
        for name, param in self.text_encoder.named_parameters():
            # Extract the layer index from parameter name
            if any(f'layer.{i}' in name for i in tune_bert_layers):
                param.requires_grad = True  # Keep trainable
            else:
                param.requires_grad = False  # Freeze
                
    def forward(self, data, codex_features):
        # Codex projection
        image_features = self.codex_projection(codex_features)

        text_features = self.text_projection(self.text_encoder(data['text'], data['att_mask']).last_hidden_state[:, 0, :])  # Use the CLS token output

        return {'codex': image_features, 'text': text_features}


class CodexCLIP_CVIT(BaseCodexCLIP):
    def __init__(self, marker_groups, device='cpu', pt_path='', **kwargs):
        super(CodexCLIP_CVIT, self).__init__(**kwargs)
        
        # Initialize the CodexEncoder
        self.codex_encoder = models.encoders.CodexEncoder(marker_groups=marker_groups, device=device, pt_path=pt_path)
        for param in self.codex_encoder.parameters():
            param.requires_grad = True

    def forward(self, data):
        # Extract codex features using the CodexEncoder
        codex_features = self.codex_encoder(data['codex'])
        return super().forward(data, codex_features)


class CodexCLIP_CNN(BaseCodexCLIP):
    def __init__(self, vocab=[], embed_dim=768, nhead=6, num_layers=12,  **kwargs):
        super(CodexCLIP_CNN, self).__init__(**kwargs)
        
        # Initialize the CodexCNNTransformer
        self.codex_encoder = models.encoders.CodexCNNTransformer(vocab=vocab, embed_dim=embed_dim, nhead=nhead, num_layers=num_layers)
        for param in self.codex_encoder.parameters():
            param.requires_grad = True

    def forward(self, data):
        # Extract codex features using the CodexCNNTransformer
        codex_features = self.codex_encoder(data['codex'], data['channels'])
        return super().forward(data, codex_features)