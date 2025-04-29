from argparse import Namespace
import argparse


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser(
        description="LT2326 H21 Mohamed's Project")

    parser.add_argument("--dataset_dir",
                        type=str,
                        #default="/srv/data/guszarzmo/mlproject/data/mscoco_h5",
                        default="./outputs",
                        help="Directory contains processed MS COCO dataset.")

    parser.add_argument("--save_dir",
                        type=str,
                        default="./inference_output",
                        help="Directory to save the output files.")

    parser.add_argument("--config_path",
                        type=str,
                        default="config/config_cnn_test.yaml",
                        help="Path for the configuration json file.")

    parser.add_argument("--checkpoint_name",
                        type=str,
                        # default="2504.2349/checkpoint_best.pth.tar",
                        default="2704.0122/checkpoint_best.pth.tar", 
                        help="Path for the configuration json file.")

    parser.add_argument(
        '--device',
        type=str,
        default="gpu",  # gpu, cpu, mgpu
        help='path to pre-trained word Embedding.')
        
    parser.add_argument(
        '--disable_wandb',
        action='store_true',
        help='Disable Weights & Biases logging')
        
    parser.add_argument(
        '--use_val_data',
        action='store_true',
        help='Use validation data instead of test data (for debugging)')
        
    parser.add_argument(
        '--max_debug_samples',
        type=int,
        default=100,
        help='Maximum number of samples to use in debug mode')

    args = parser.parse_args()

    return args
