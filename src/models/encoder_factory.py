from torch import nn
from .cnn_encoder import ImageEncoder
from .hierarchical_encoder import EnhancedHierarchicalEncoder

def create_encoder(config):
    """
    Factory function to create the appropriate encoder based on config
    
    Config options:
    - encoder_type: Which type of encoder to use
      - "cnn": Original CNN encoder using ResNet
      - "hierarchical_cnn_vit": Enhanced encoder with hierarchical features and transformer
    
    - image_encoder: Parameters for CNN encoder
      - encode_size: Size of output feature grid (e.g., 14 for 14x14 grid)
      - embed_dim: Feature dimension
    
    - hierarchical_encoder: Parameters for hierarchical encoder
      - encode_size: Size of output feature grid
      - embed_dim: Feature dimension
      - num_heads: Number of attention heads
      - depth: Number of transformer layers
      - dropout: Dropout rate
      - use_efficientnet: True to use EfficientNet, False for ResNet
      - backbone_name: Specific backbone model (efficientnet_b3, resnet101, etc.)
    """
    encoder_type = config["hyperparams"].get("encoder_type", "cnn")
    
    if encoder_type == "cnn":
        # Create the original CNN encoder
        image_enc_params = config["hyperparams"]["image_encoder"]
        return ImageEncoder(**image_enc_params)
    
    elif encoder_type == "hierarchical_cnn_vit":
        # Create the hierarchical CNN-ViT encoder
        hier_params = config["hyperparams"]["hierarchical_encoder"]
        return EnhancedHierarchicalEncoder(**hier_params)
    
    else:
        valid_types = ["cnn", "hierarchical_cnn_vit"]
        raise ValueError(f"Unknown encoder type: {encoder_type}. Supported types: {valid_types}")