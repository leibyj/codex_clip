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
        
        self.text_encoder = BertModel.from_pretrained(hf_model)
       
        if freeze_bert_layers and tune_bert_layers is not None:
            self._freeze_bert_layers(tune_bert_layers)
        else:
            for param in self.text_encoder.parameters():
                param.requires_grad = True
                
        # shared or independent projection layers
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


class BimodalEncoder(nn.Module):
    def __init__(self, 
                 modalities=['codex', 'text'],
                 hf_model="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                 codex_dim=768,
                 text_dim=768,
                 he_dim=768,
                 projection_dim=512,
                 shared_projection=False,
                 freeze_bert_layers=False,
                 tune_bert_layers=None,
                 vocab=[], 
                 nhead=6, 
                 num_layers=12):
        """
        Args:
            modalities (list): A list of two modalities to be used. Each element is one of 'codex', 'he', 'text'.
            hf_model (str): The Hugging Face model identifier for the text encoder.
            codex_dim (int): The dimensionality of the Codex encoder output.
            text_dim (int): The dimensionality of the text encoder output.
            he_dim (int): The dimensionality of the HandE encoder output.
            projection_dim (int): The dimensionality of the projection layers.
            shared_projection (bool): If True, a shared projection layer is used for both modalities.
            freeze_bert_layers (bool): If True, freezes the BERT layers in the text encoder.
            tune_bert_layers (int or None): Number of top BERT layers to fine-tune.
            vocab (list): Vocabulary for the CodexCNNTransformer.
            nhead (int): Number of attention heads for the CodexCNNTransformer.
            num_layers (int): Number of layers for the CodexCNNTransformer.
        """
        super(BimodalEncoder, self).__init__()

        if not isinstance(modalities, list) or len(modalities) != 2:
            raise ValueError("modalities must be a list of two modalities.")

        self.modalities = modalities

        # Initialize encoders and projections for the specified modalities
        for modality in modalities:
            if modality == 'text':
                self.text_encoder = models.encoders.TextEncoder(hf_model, freeze_bert_layers, tune_bert_layers)
            elif modality == 'codex':
                self.codex_encoder = models.encoders.CodexCNNTransformer(vocab=vocab, embed_dim=codex_dim, nhead=nhead, num_layers=num_layers)
                for param in self.codex_encoder.parameters():
                    param.requires_grad = True
            elif modality == 'he':
                self.he_encoder = models.encoders.PLIPEncoder()
                for param in self.he_encoder.parameters():
                    param.requires_grad = True
            else:
                raise ValueError(f"Unknown modality: {modality}")

        # Handle shared or independent projections
        if shared_projection:
            dims = []
            for modality in modalities:
                if modality == 'text':
                    dims.append(text_dim)
                elif modality == 'codex':
                    dims.append(codex_dim)
                elif modality == 'he':
                    dims.append(he_dim)
            if len(set(dims)) == 1:
                shared = nn.Linear(dims[0], projection_dim)
                for modality in modalities:
                    setattr(self, f"{modality}_projection", shared)
            else:
                # Map modalities to common dim, then shared projection
                common_dim = max(dims)
                shared = nn.Linear(common_dim, projection_dim)
                for modality, dim in zip(modalities, dims):
                    if dim != common_dim:
                        proj = nn.Sequential(
                            nn.Linear(dim, common_dim),
                            nn.GELU(),
                            nn.Dropout(0.1),
                            nn.LayerNorm(common_dim),
                            shared
                        )
                    else:
                        proj = shared
                    setattr(self, f"{modality}_projection", proj)
        else:
            # Independent projections
            for modality in modalities:
                if modality == 'text':
                    proj = nn.Linear(text_dim, projection_dim)
                elif modality == 'codex':
                    proj = nn.Linear(codex_dim, projection_dim)
                elif modality == 'he':
                    proj = nn.Linear(he_dim, projection_dim)
                setattr(self, f"{modality}_projection", proj)

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

    def get_features_single_modality(self, x, modality):
        self.eval()

        if not isinstance(modality, str) or modality not in self.modalities:
            raise ValueError(f"Modality must be one of {self.modalities}")

        encoder = getattr(self, f"{modality}_encoder")
        projection = getattr(self, f"{modality}_projection")

        if modality == 'text':
            out = projection(encoder(x['text'], x['att_mask']))
        elif modality == 'codex':
            out = projection(encoder(x['codex'], x['channels']))
        elif modality == 'he':
            out = projection(encoder(x['HandE']))
        else:
            raise ValueError(f"Unknown modality: {modality}")

        return out

    def forward(self, data):
        outputs = {}
        for modality in self.modalities:
            encoder = getattr(self, f"{modality}_encoder")
            projection = getattr(self, f"{modality}_projection")

            if modality == 'text':
                features = encoder(data['text'], data['att_mask'])
            elif modality == 'codex':
                features = encoder(data['codex'], data['channels'])
            elif modality == 'he':
                features = encoder(data['HandE'])
            else:
                raise ValueError(f"Unknown modality: {modality}")

            projected_features = projection(features)
            outputs[modality] = projected_features

        return outputs
    
class TrimodalEncoder(nn.Module):
    def __init__(self, 
                 hf_model="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                 codex_dim=768,
                 text_dim=768,
                 he_dim=768,
                 projection_dim=512,
                 shared_projection=False,
                 freeze_bert_layers=False,
                 tune_bert_layers=None,
                 vocab=[], 
                 nhead=6, 
                 num_layers=12):
        """
        Args:
            hf_model (str): The Hugging Face model identifier for the text encoder.
            codex_dim (int): The dimensionality of the Codex encoder output.
            text_dim (int): The dimensionality of the text encoder output.
            he_dim (int): The dimensionality of the HandE encoder output.
            projection_dim (int): The dimensionality of the projection layers.
            shared_projection (bool): If True, a shared projection layer is used for both Codex and text features.
            vocab (list): Vocabulary for the CodexCNNTransformer.
            nhead (int): Number of attention heads for the CodexCNNTransformer.
            num_layers (int): Number of layers for the CodexCNNTransformer.
        """
        super(TrimodalEncoder, self).__init__()
        
        self.text_encoder = models.encoders.TextEncoder(hf_model, freeze_bert_layers, tune_bert_layers)
       
        self.codex_encoder = models.encoders.CodexCNNTransformer(vocab=vocab, embed_dim=codex_dim, nhead=nhead, num_layers=num_layers)
        for param in self.codex_encoder.parameters():
            param.requires_grad = True
        
        self.he_encoder = models.encoders.PLIPEncoder()
        for param in self.he_encoder.parameters():
            param.requires_grad = True

        # shared or independent projection layers
        if shared_projection:
            if codex_dim == text_dim == he_dim:
                shared = nn.Linear(codex_dim, projection_dim)
                self.codex_projection = shared
                self.text_projection = shared
                self.he_projection = shared
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
                self.he_projection = nn.Sequential(
                    nn.Linear(he_dim, text_dim),
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
            self.he_projection = nn.Linear(he_dim, projection_dim)
        
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

    def get_features_single_modality(self, x, modality='he'):
        self.eval()

        if not isinstance(modality, str) or modality not in ['he', 'codex', 'text']:
            raise ValueError("Modality must be one of ['he', 'codex', 'text']")

        encoder = getattr(self, f"{modality}_encoder")
        projection = getattr(self, f"{modality}_projection")

        if modality == 'text':
            out = projection(encoder(x['text'], x['att_mask']))
        elif modality == 'codex':
            out = projection(encoder(x['codex'], x['channels']))
        elif modality == 'he':
            out = projection(encoder(x['HandE']))
        else:
            raise ValueError(f"Unknown modality: {modality}")

        return out


    def forward(self, data):
        # Extract codex features using the CodexCNNTransformer
        codex_features = self.codex_encoder(data['codex'], data['channels'])
        he_features = self.he_encoder(data['HandE'])
        text_features = self.text_encoder(data['text'], data['att_mask'])

        codex_features = self.codex_projection(codex_features)
        he_features = self.he_projection(he_features)   
        text_features = self.text_projection(text_features)

        return {'codex': codex_features, 'text': text_features, "HandE": he_features}