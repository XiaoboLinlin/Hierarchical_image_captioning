#!/usr/bin/env python
"""
Utility script to fix checkpoint dimension mismatches by adding projection dimensions
to config or modifying the checkpoint to work with a different backbone.
"""

import os
import argparse
import torch
from pathlib import Path
import yaml
import copy


def parse_args():
    parser = argparse.ArgumentParser(description="Fix checkpoint dimension mismatches")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--output_config", type=str, help="Path to save the modified config file (optional)")
    parser.add_argument("--output_checkpoint", type=str, help="Path to save the modified checkpoint file (optional)")
    parser.add_argument("--target_backbone", type=str, default=None, 
                       help="Target backbone model (efficientnet_b3, efficientnet_b4, etc.)")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    return parser.parse_args()


def detect_backbone_from_checkpoint(checkpoint_path, debug=False):
    """Detect the backbone model used in the checkpoint by analyzing layer shapes"""
    state = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
    model_state = state["models"][0]  # First model is the encoder
    
    # Check for specific layer dimensions to identify the backbone
    backbone_dims = {
        "efficientnet_b3": {"proj0": 40, "proj1": 112, "proj2": 384},
        "efficientnet_b4": {"proj0": 48, "proj1": 136, "proj2": 448},
        "efficientnet_b0": {"proj0": 24, "proj1": 80, "proj2": 320}
    }
    
    # Try to find projection layer dimensions
    proj_dims = []
    for i in range(3):  # Check the first 3 projection layers
        key = f"encoder.projections.{i}.weight"
        if key in model_state:
            dim = model_state[key].shape[1]
            proj_dims.append(dim)
            if debug:
                print(f"Found projection layer {i} with input dimension: {dim}")
    
    if not proj_dims:
        print("Warning: Could not find projection layers in the checkpoint")
        return None, []
    
    # Find the best matching backbone
    best_match = None
    best_score = 0
    for backbone, dims in backbone_dims.items():
        score = sum(1 for i, dim in enumerate(proj_dims) if i < len(dims) and dim == dims[f"proj{i}"])
        if score > best_score:
            best_score = score
            best_match = backbone
    
    if debug:
        print(f"Detected projection dimensions: {proj_dims}")
        print(f"Best matching backbone: {best_match} (score: {best_score}/{len(proj_dims)})")
    
    return best_match, proj_dims


def modify_config_for_checkpoint(config_path, backbone_name, proj_dims, output_path=None, debug=False):
    """Modify the config file to work with the checkpoint"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Make a copy of the original config
    original_config = copy.deepcopy(config)
    
    # Update backbone name and add projection dimensions
    if backbone_name:
        config["hyperparams"]["hierarchical_encoder"]["backbone_name"] = backbone_name
    
    if proj_dims:
        config["hyperparams"]["hierarchical_encoder"]["projection_dims"] = proj_dims
    
    # Save the modified config if output path is provided
    if output_path:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Modified config saved to {output_path}")
    
    if debug:
        print(f"Original backbone: {original_config['hyperparams']['hierarchical_encoder'].get('backbone_name', 'unknown')}")
        print(f"New backbone: {config['hyperparams']['hierarchical_encoder']['backbone_name']}")
        print(f"Added projection dimensions: {proj_dims}")
    
    return config


def main():
    args = parse_args()
    
    # Ensure paths exist
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file {checkpoint_path} does not exist")
        return
    
    if not config_path.exists():
        print(f"Error: Config file {config_path} does not exist")
        return
    
    # Determine output paths
    output_config = args.output_config or str(config_path).replace('.yaml', '_modified.yaml')
    output_checkpoint = args.output_checkpoint
    
    # Detect backbone from checkpoint
    detected_backbone, proj_dims = detect_backbone_from_checkpoint(
        str(checkpoint_path), debug=args.debug
    )
    
    print(f"Detected backbone: {detected_backbone}")
    print(f"Detected projection dimensions: {proj_dims}")
    
    # Use the target backbone if specified, otherwise use the detected one
    backbone = args.target_backbone or detected_backbone
    
    # Modify the config
    modified_config = modify_config_for_checkpoint(
        str(config_path), backbone, proj_dims, output_config, debug=args.debug
    )
    
    print("\nConfig modification completed.")
    print(f"Modified config saved to: {output_config}")
    print("\nTo use this config with the checkpoint:")
    print(f"python src/inference_vit_cnn.py --config_path {output_config} --checkpoint_name {os.path.basename(args.checkpoint)}")


if __name__ == "__main__":
    main() 