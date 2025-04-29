import yaml

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def validate_config(config):
    """Validates configuration and provides helpful error messages"""
    
    # Check encoder type
    encoder_type = config["hyperparams"].get("encoder_type")
    valid_encoder_types = ["cnn", "hierarchical_cnn_vit"]
    
    if encoder_type not in valid_encoder_types:
        raise ValueError(f"Invalid encoder_type: {encoder_type}. Must be one of: {valid_encoder_types}")
    
    # Check backbone if using hierarchical
    if encoder_type == "hierarchical_cnn_vit":
        backbone = config["hyperparams"]["hierarchical_encoder"].get("backbone_name")
        valid_backbones = ["resnet101", "efficientnet_b0", "efficientnet_b3", "efficientnet_b4"]
        
        if backbone not in valid_backbones:
            raise ValueError(f"Invalid backbone_name: {backbone}. Must be one of: {valid_backbones}")
    
    return True