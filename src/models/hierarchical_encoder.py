import torch
from torch import nn, Tensor
import torchvision
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

class CrossScaleAttention(nn.Module):
    """Cross-scale attention module to connect features from different scales"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(CrossScaleAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        """
        Args:
            query: Features at current scale [seq_len, batch, embed_dim]
            key_value: Features from another scale [seq_len, batch, embed_dim]
        Returns:
            Enhanced features with cross-scale information
        """
        attn_output, _ = self.multihead_attn(
            query=query,
            key=key_value,
            value=key_value
        )
        return self.norm(query + self.dropout(attn_output))

class EnhancedHierarchicalEncoder(nn.Module):
    def __init__(self, 
                 encode_size=14, 
                 embed_dim=512, 
                 num_heads=8, 
                 depth=6, 
                 dropout=0.1,
                 use_efficientnet=True,
                 backbone_name="efficientnet_b3",
                 projection_dims=None):
        super(EnhancedHierarchicalEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.encode_size = encode_size
        self.handle_raw_images = True
        
        # Select backbone based on configuration
        if use_efficientnet:
            if backbone_name == "efficientnet_b3":
                self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
                # EfficientNet-B3 feature dimensions at different blocks
                self.feature_layers = [3, 5, 7]  # Corresponds to different resolution blocks
                feat_dims = [40, 112, 384]  # Corresponding to the feature_layers
            elif backbone_name == "efficientnet_b0":
                self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
                self.feature_layers = [3, 5, 7]
                feat_dims = [24, 80, 320]
            elif backbone_name == "efficientnet_b4":
                self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
                self.feature_layers = [3, 5, 7]
                feat_dims = [48, 136, 448]  # EfficientNet-B4 dimensions
            else:
                raise ValueError(f"Unsupported EfficientNet variant: {backbone_name}")
        else:
            # Use ResNet backbone
            self.backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            # For ResNet, we'll need a different approach to extract features
            self.feature_extractor = "resnet"
            # ResNet feature dimensions
            feat_dims = [256, 512, 2048]  # From different ResNet blocks
        
        # Override feature dimensions with custom projection dims if provided
        if projection_dims is not None:
            if len(projection_dims) >= 2:
                feat_dims[:len(projection_dims)] = projection_dims
                print(f"Using custom projection dimensions: {projection_dims}")
        
        # Feature projections for each scale
        self.projections = nn.ModuleList([
            nn.Conv2d(dim, embed_dim, kernel_size=1, stride=1, bias=False) 
            for dim in feat_dims
        ])
        
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(embed_dim) for _ in range(len(feat_dims))
        ])
        
        self.relu = nn.ReLU(inplace=True)
        
        # Adaptive pooling to get fixed size at each scale
        self.adaptive_resizers = nn.ModuleList([
            nn.AdaptiveAvgPool2d((encode_size, encode_size)) for _ in range(len(feat_dims))
        ])
        
        # Position embeddings for transformer (one for each scale)
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, encode_size * encode_size, embed_dim)) 
            for _ in range(len(feat_dims))
        ])
        
        # Transformer encoder layers for each scale
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True
        )
        
        self.transformers = nn.ModuleList([
            nn.TransformerEncoder(encoder_layer, num_layers=depth//2)  # Smaller depth for individual scales
            for _ in range(len(feat_dims))
        ])
        
        # Cross-scale attention modules for feature fusion (new addition)
        self.cross_scale_attentions = nn.ModuleList([
            CrossScaleAttention(embed_dim, num_heads//2, dropout)
            for _ in range(len(feat_dims) - 1)  # One less than number of scales
        ])
        
        # Scale mixing layers with residual connections (new addition)
        self.scale_mixers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(len(feat_dims) - 1)
        ])
        
        # Final fusion transformer
        fusion_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(fusion_encoder_layer, num_layers=depth)
        
        # Scale tokens to help fusion transformer distinguish features
        self.scale_tokens = nn.Parameter(torch.zeros(len(feat_dims), 1, embed_dim))
        
        # Global feature integration (new addition)
        self.global_feature_gate = nn.Sequential(
            nn.Linear(embed_dim * len(feat_dims), embed_dim),
            nn.Sigmoid()
        )
        
        self.global_feature_transform = nn.Sequential(
            nn.Linear(embed_dim * len(feat_dims), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize position embeddings
        for pos_embed in self.pos_embeds:
            nn.init.trunc_normal_(pos_embed, std=0.02)
        
        # Initialize scale tokens
        nn.init.trunc_normal_(self.scale_tokens, std=0.02)
        
    def _extract_efficientnet_features(self, x):
        features = []
        
        # Process through the backbone one layer at a time
        for idx, module in enumerate(self.backbone.features):
            x = module(x)
            if idx in self.feature_layers:
                features.append(x)
                
        return features
    
    def _extract_resnet_features(self, x):
        features = []
        
        # Get features from different ResNet blocks
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        features.append(x)  # First feature map
        
        x = self.backbone.layer2(x)
        features.append(x)  # Second feature map
        
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        features.append(x)  # Third feature map
        
        return features
    
    def forward(self, images):
        """
        params:
        images: Input images.
                Tensor [batch_size, 3, h, w]

        output: encoded images.
                Tensor [batch_size, encode_size * encode_size, embed_dim]
        """
        # Extract features at different scales
        if hasattr(self, 'feature_extractor') and self.feature_extractor == "resnet":
            features = self._extract_resnet_features(images)
        else:
            features = self._extract_efficientnet_features(images)
            
        # Ensure projections match the actual feature dimensions
        for i, feat in enumerate(features):
            if feat.size(1) != self.projections[i].in_channels:
                self.projections[i] = nn.Conv2d(
                    in_channels=feat.size(1),
                    out_channels=self.embed_dim,
                    kernel_size=1,
                    stride=1,
                    bias=False
                ).to(feat.device)
        
        # Process features from each scale
        multi_scale_features = []
        
        for i, feat in enumerate(features):
            # 1. Project features to common dimension
            x = self.relu(self.bns[i](self.projections[i](feat)))
            
            # 2. Resize to fixed grid size
            x = self.adaptive_resizers[i](x)
            
            # 3. Flatten spatial dimensions and add positional embeddings
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
            x = x + self.pos_embeds[i]
            
            # 4. Apply transformer to capture contextual info
            x = self.transformers[i](x)
            
            multi_scale_features.append(x)
        
        # Cross-scale attention (optional feature fusion)
        enhanced_features = []
        enhanced_features.append(multi_scale_features[0])  # Start with finest scale
        
        # Apply cross-scale attention between adjacent scales
        for i in range(len(multi_scale_features) - 1):
            current = multi_scale_features[i]
            next_scale = multi_scale_features[i+1]
            
            # Convert to sequence-first for attention
            current_seq = current.transpose(0, 1)  # [S, B, D]
            next_seq = next_scale.transpose(0, 1)  # [S, B, D]
            
            # Apply cross attention
            fused = self.cross_scale_attentions[i](current_seq, next_seq)
            
            # Convert back to batch-first
            fused = fused.transpose(0, 1)  # [B, S, D]
            
            # Concatenate and mix features
            concat_feats = torch.cat([fused, next_scale], dim=-1)  # [B, S, D*2]
            mixed_feats = self.scale_mixers[i](concat_feats) + next_scale  # [B, S, D]
            
            enhanced_features.append(mixed_feats)
        
        # Fusion stage for all scales
        # Add scale tokens to distinguish features
        scale_feats = []
        for i, feats in enumerate(enhanced_features):
            # Add scale token to each sequence position
            scale_token = self.scale_tokens[i].expand(feats.size(0), feats.size(1), -1)
            scale_feats.append(feats + scale_token)
            
        # Concatenate along sequence dimension
        fused_feats = torch.cat(scale_feats, dim=1)  # [B, S*scales, D]
        
        # Apply final transformer for global reasoning
        final_features = self.fusion_transformer(fused_feats)
        
        # Global feature integration
        # Extract a sequence-level representation for gating
        seq_lens = [feat.size(1) for feat in scale_feats]
        splits = torch.split(final_features, seq_lens, dim=1)
        
        # Compute mean representations of each scale
        means = [split.mean(dim=1, keepdim=True) for split in splits]  # [[B, 1, D], ...]
        
        # Concatenate means and compute gating weights
        concat_means = torch.cat([m.squeeze(1) for m in means], dim=-1)  # [B, D*scales]
        gate = self.global_feature_gate(concat_means).unsqueeze(1)  # [B, 1, D]
        
        # Transform the concatenated features
        global_feat = self.global_feature_transform(concat_means).unsqueeze(1)  # [B, 1, D]
        
        # Final feature combination - only use first scale features
        output = scale_feats[0] * gate + global_feat * (1 - gate)
        
        return output  # [B, S, D]