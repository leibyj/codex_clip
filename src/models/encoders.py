
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CodexEncoder(nn.Module):
    def __init__(self, marker_groups=0, hidden_dim=384, device='cpu', pt_path=''):
        """
        Args:
            marker_groups (int): The number of marker groups, determining how many ChannelViTs to initialize.
            hidden_dim (int): The dimension to project the combined embeddings into, also used for self-attention.
            device (str): The device to load ChannelViTs
        """
        super(CodexEncoder, self).__init__()

        # initialize ChannelViTs for each marker group        
        self.channelvit = nn.moduleList([self._load_channelvit_model(device, pt_path) for i in range(marker_groups)])

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


    def _load_channelvit_model(self, device, pt_path):
        # trouble getting it to load on correct device without adding that here
        model = torch.hub.load('insitro/ChannelViT', 'cpjump_cellpaint_bf_channelvit_small_p8_with_hcs_supervised', pretrained=False)
        state_dict = torch.load(f"{pt_path}/cpjump_cellpaint_bf_channelvit_small_p8_with_hcs_supervised.pth", map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        return model.to(device)
    
    def forward(self, x):

        cls_tokens = []
        for i, dat in enumerate(x):
            cls_tokens.append(self.channelvit[i](dat[0], dat[1]))

        # Concatenate CLS embeddings along the feature dimension and project to CA embedding size
        combined_cls = torch.cat(cls_tokens, dim=1)  # Shape: (batch_size, maker_group * embedding_dim)
        combined_cls = self.projection(combined_cls)
        # Reshape for self attention (needs 3D tensor: (sequence_length, batch_size, embed_dim))
        combined_cls = combined_cls.unsqueeze(0)  # Shape: (1, batch_size, embedding_dim)

        # # Perform self attention
        attn_output, _ = self.self_attention(combined_cls, combined_cls, combined_cls)
        attn_output = self.attn_postprocess(attn_output)

        return attn_output.squeeze(0)


class ResNetbackbone(nn.Module):
    def __init__(self, embed_dim):
        """
        Args:
            embed_dim (int): The dimensionality of the output embedding.
        """
        super(ResNetbackbone, self).__init__()
        # Load the pretrained ResNet
        resnet = models.resnet18(pretrained=True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
        resnet.conv1 = self.conv1
        
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        
        self.fc = nn.Linear(resnet.fc.in_features, embed_dim)

    def forward(self, x):
        out = self.resnet(x)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1)) 
        out = out.view(out.size(0), -1)  
        out = self.fc(out) 
        return out

class CNNbackbone(nn.Module):
    def __init__(self, embed_dim):
        """
        Args:
            embed_dim (int): The dimensionality of the output embedding.
        """
        super(CNNbackbone, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Fully connected layer to create the final embedding of size embed_dim
        self.fc = nn.Linear(128 * (256 // 8) * (256 // 8), embed_dim)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ChannelEmbedding(nn.Module):
    def __init__(self, embedding_dim, vocab):
        """
        Args:
            embedding_dim (int): Size of the embeddings for each channel.
            vocab (list of str): List of all possible strings corresponding to channel values.
        """
        super(ChannelEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.channel_embedding = nn.Embedding(len(vocab), embedding_dim)
        self.channel_to_idx = {v: i for i, v in enumerate(vocab)}

    def forward(self, channel_names):
        """
        Args:
            channel_names (list of list of str): Batch of lists of strings corresponding to channels in each sample
        """
        max_c = max(len(c) for c in channel_names)
        batch_embeddings = []
        padding_mask = []
        
        for channels in channel_names:
            c_indices = [self.channel_to_idx[c] for c in channels]
            c_indices_tensor = torch.tensor(c_indices, device=self.channel_embedding.weight.device)
            embeddings = self.channel_embedding(c_indices_tensor)

            padding_len = max_c - len(channels)
            if padding_len > 0:
                embeddings = F.pad(embeddings, (0, 0, 0, padding_len), value=0)

            batch_embeddings.append(embeddings)
            padding_mask.append([0] * len(channels) + [1] * padding_len)

        batch_embeddings = torch.stack(batch_embeddings, dim=0)
        padding_mask = torch.tensor(padding_mask, device=batch_embeddings.device, dtype=torch.bool)


        return batch_embeddings, padding_mask

class CodexCNNTransformer(nn.Module):
    def __init__(self, vocab=[], embed_dim=768, nhead=6, num_layers=12):
        """
        Args:
            vocab (list of str): List of all possible channel names.
            embed_dim (int): Feature embedding size.
            nhead (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
        """
        super(CodexCNNTransformer, self).__init__()
        # self.cnn = CNNbackbone(embed_dim)
        self.cnn = ResNetbackbone(embed_dim)
        self.channel_embedding = ChannelEmbedding(embed_dim, vocab)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # start from random...?

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, samples, c_strings):
        """
        Args:
            samples (Tensor): Batch of image tensors of shape (batch_size, c, 256, 256).
            c_strings (list of list of str): Batch of lists of strings of channels for each sample
        """
        batch_size, num_channels, _, _ = samples.size()
        
        chn_imgs = samples.view(batch_size * num_channels, 1, 256, 256)
        cnn_embeddings = self.cnn(chn_imgs)
        cnn_embeddings = cnn_embeddings.view(batch_size, num_channels, -1)  # Reshape back to (batch_size, num_channels, embed_dim)

        # channel embeddings and CLS token
        channel_embeddings, padding_mask = self.channel_embedding(c_strings)
        # print(channel_embeddings.shape)
        x = cnn_embeddings + channel_embeddings  #  Add or concatenate?
        batch_cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat([batch_cls_token, x], dim=1) 
        cls_padding_mask = torch.zeros(batch_size, 1, device=x.device, dtype=torch.bool)
        padding_mask = torch.cat([cls_padding_mask, padding_mask], dim=1)  # (batch_size, num_channels + 1)

        out = self.transformer_encoder(x.transpose(0, 1), src_key_padding_mask=padding_mask)  # (num_channels + 1, batch_size, embed_dim)

        cls_embedding = out[0]  # (1, batch_size, embed_dim)
        return cls_embedding.squeeze(0) 

class CodexCNNTransformerIBOT(nn.Module):
    def __init__(self, vocab=[], embed_dim=768, nhead=6, num_layers=12, mask_ratio=0.75, teacher_temperature=0.07, student_temperature=0.1):
        """
        Args:
            vocab (list of str): List of all possible channel names.
            embed_dim (int): Feature embedding size.
            nhead (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            mask_ratio (float): Ratio of patches to mask in each input image.
            teacher_temperature (float): Temperature parameter for teacher network's softmax.
        """
        super(CodexCNNTransformerIBOT, self).__init__()
        self.cnn = ResNetbackbone(embed_dim)
        self.channel_embedding = ChannelEmbedding(embed_dim, vocab)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # CLS token for global feature representation
        self.mask_ratio = mask_ratio
        self.teacher_temperature = teacher_temperature
        self.student_temperature = student_temperature

        # student
        student_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(student_encoder_layer, num_layers=num_layers)
        self.student_projection = nn.Linear(embed_dim, embed_dim)
        self._initialize_weights()

        # teacher
        teacher_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.teacher = nn.TransformerEncoder(teacher_encoder_layer, num_layers=num_layers)
        self.teacher_projection = nn.Linear(embed_dim, embed_dim)

        self.teacher.load_state_dict(self.transformer_encoder.state_dict())
        self.teacher_projection.load_state_dict(self.student_projection.state_dict())

        for param in self.teacher.parameters():
            param.requires_grad = False
        for param in self.teacher_projection.parameters():
            param.requires_grad = False

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def random_masking(self, x, mask_ratio):
        """
        Randomly mask a portion of the input embeddings.
        Args:
            x (Tensor): Input embeddings (batch_size, num_channels, embed_dim).
            mask_ratio (float): The percentage of embeddings to mask.
        """
        batch_size, num_patches, _ = x.size()
        num_masked = int(num_patches * mask_ratio)

        # Generate random indices to mask
        indices = torch.rand(batch_size, num_patches, device=x.device).argsort(dim=1)
        masked_indices = indices[:, :num_masked]
        visible_indices = indices[:, num_masked:]

        mask = torch.ones_like(x)
        for i in range(batch_size):
            mask[i, masked_indices[i]] = 0  # Masked patches

        return x * mask, masked_indices, visible_indices

    def forward(self, samples, c_strings):
        """
        Args:
            samples (Tensor): Batch of image tensors of shape (batch_size, c, 256, 256).
            c_strings (list of list of str): Batch of lists of strings of channels for each sample
        """
        batch_size, num_channels, _, _ = samples.size()

        chn_imgs = samples.view(batch_size * num_channels, 1, 256, 256)
        cnn_embeddings = self.cnn(chn_imgs)
        cnn_embeddings = cnn_embeddings.view(batch_size, num_channels, -1)  # Reshape to (batch_size, num_channels, embed_dim)

        # Channel embeddings, masking, and cls token
        channel_embeddings, padding_mask = self.channel_embedding(c_strings)
        combined_embeddings = cnn_embeddings + channel_embeddings
        combined_masked, masked_indices, visible_indices = self.random_masking(combined_embeddings, self.mask_ratio)
        batch_cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat([batch_cls_token, combined_masked], dim=1)  # (batch_size, num_patches + 1, embed_dim)
        cls_padding_mask = torch.zeros(batch_size, 1, device=x.device, dtype=torch.bool)
        padding_mask = torch.cat([cls_padding_mask, padding_mask], dim=1)  # (batch_size, num_patches + 1)

        # student
        out = self.transformer_encoder(x.transpose(0, 1), src_key_padding_mask=padding_mask)  # (num_patches + 1, batch_size, embed_dim)
        student_pred = self.student_projection(out.transpose(0, 1))  # (batch_size, num_patches + 1, embed_dim)

        # teacher
        with torch.no_grad():
            teacher_out = self.teacher(x.transpose(0, 1), src_key_padding_mask=padding_mask)
            teacher_pred = self.teacher_projection(teacher_out.transpose(0, 1))  # (batch_size, num_patches + 1, embed_dim)
        
        student_pred = F.normalize(student_pred, dim=-1)
        teacher_pred = F.normalize(teacher_pred, dim=-1)

        loss = self.compute_ibot_loss(student_pred, teacher_pred, masked_indices)
        return loss


    def compute_ibot_loss(self, student_pred, teacher_pred, masked_indices):
        batch_size, num_patches, _ = student_pred.size()

        # Select only the masked tokens
        student_masked_pred = torch.stack([student_pred[i, masked_indices[i]] for i in range(batch_size)])
        teacher_masked_pred = torch.stack([teacher_pred[i, masked_indices[i]] for i in range(batch_size)])

        # Apply temperature scaling and compute probabilities
        teacher_probs = F.softmax(teacher_masked_pred / self.teacher_temperature, dim=-1)
        student_log_probs = F.log_softmax(student_masked_pred / self.student_temperature, dim=-1)

        # Compute KL divergence loss
        loss_fn = nn.KLDivLoss(reduction='batchmean')
        loss = loss_fn(student_log_probs, teacher_probs)

        return loss


    def update_teacher(self, momentum=0.996):
        for param_s, param_t in zip(self.transformer_encoder.parameters(), self.teacher.parameters()):
            param_t.data = momentum * param_t.data + (1 - momentum) * param_s.data

        for param_s, param_t in zip(self.student_projection.parameters(), self.teacher_projection.parameters()):
            param_t.data = momentum * param_t.data + (1 - momentum) * param_s.data