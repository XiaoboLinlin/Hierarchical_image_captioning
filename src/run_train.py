from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from models.encoder_factory import create_encoder
from models.IC_encoder_decoder.transformer import Transformer

from dataset.dataloader import HDF5Dataset, collate_padd
from torchtext.vocab import Vocab

from trainer import Trainer
from utils.train_utils import parse_arguments, seed_everything
from utils.config_loader import load_config, validate_config
from utils.gpu_cuda_helper import select_device

# Other imports


def get_datasets(dataset_dir: str, pid_pad: float):
    # Setting some pathes
    dataset_dir = Path(dataset_dir)
    images_train_path = dataset_dir / "train_images.hdf5"
    images_val_path = dataset_dir / "val_images.hdf5"
    captions_train_path = dataset_dir / "train_captions.json"
    captions_val_path = dataset_dir / "val_captions.json"
    lengthes_train_path = dataset_dir / "train_lengthes.json"
    lengthes_val_path = dataset_dir / "val_lengthes.json"

    # images transfrom
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = Compose([norm])

    train_dataset = HDF5Dataset(hdf5_path=images_train_path,
                                captions_path=captions_train_path,
                                lengthes_path=lengthes_train_path,
                                pad_id=pid_pad,
                                transform=transform)

    val_dataset = HDF5Dataset(hdf5_path=images_val_path,
                              captions_path=captions_val_path,
                              lengthes_path=lengthes_val_path,
                              pad_id=pid_pad,
                              transform=transform)

    return train_dataset, val_dataset

if __name__ == "__main__":
    # Parse command arguments
    args = parse_arguments()
    dataset_dir = args.dataset_dir
    resume = args.resume
    resume = None if resume == "" else resume

    # Device selection
    device = select_device(args.device, args.gpu_index)
    print(f"selected device is {device}.\n")

    # Load YAML configuration instead of JSON
    config = load_config(args.config_path)
    validate_config(config)

    # Load vocab
    min_freq = config["min_freq"]
    vocab: Vocab = torch.load(str(Path(dataset_dir) / "vocab.pth"))
    pad_id = vocab.stoi["<pad>"]
    vocab_size = len(vocab)

    # SEED
    SEED = config["seed"]
    seed_everything(SEED)

    # --------------- dataloader --------------- #
    print("loading dataset...")
    g = torch.Generator()
    g.manual_seed(SEED)
    loader_params = config["dataloader_parms"]
    max_len = config["max_len"]
    train_ds, val_ds = get_datasets(dataset_dir, pad_id)
    
    # Check for debug mode
    debug = config.get("debug", {}).get("enabled", False)
    max_debug_samples = config.get("debug", {}).get("max_samples", 1000)

    # Decide whether to wrap in a Subset
    if debug:
        from torch.utils.data import Subset
        print(f"Running in debug mode with max {max_debug_samples} samples")
        train_ds_used = Subset(train_ds, list(range(min(len(train_ds), max_debug_samples))))
        val_ds_used = Subset(val_ds, list(range(min(len(val_ds), max_debug_samples//5))))
    else:
        train_ds_used = train_ds
        val_ds_used = val_ds

    train_iter = DataLoader(
        train_ds_used,
        collate_fn=collate_padd(max_len, pad_id),
        pin_memory=True,
        **loader_params
    )

    val_iter = DataLoader(
        val_ds_used,
        collate_fn=collate_padd(max_len, pad_id),
        batch_size=loader_params['batch_size'],
        pin_memory=True,
        num_workers=1,
        shuffle=True
    )
    
    print("loading dataset finished.")
    print(f"number of vocabulary is {vocab_size}\n")

    # --------------- Construct models, optimizers --------------- #
    print("constructing models")
    
    # Get encoder type from config
    encoder_type = config["hyperparams"].get("encoder_type", "cnn")
    use_enhanced_decoder = config["hyperparams"].get("transformer", {}).get("use_enhanced_decoder", False)
    
    # Create the appropriate encoder using our factory
    encoder = create_encoder(config)
    
    # Prepare transformer hyperparameters
    transformer_hyperparms = config["hyperparams"]["transformer"]
    transformer_hyperparms["vocab_size"] = vocab_size
    transformer_hyperparms["pad_id"] = pad_id
    
    if encoder_type == "cnn":
        # For CNN, we need to set up the image dimensions
        image_seq_len = int(config["hyperparams"]["image_encoder"]["encode_size"]**2)
        transformer_hyperparms["img_encode_size"] = image_seq_len
        transformer_hyperparms["max_len"] = max_len - 1
        
        # Make sure we don't have duplicate parameters
        if "use_enhanced_decoder" in transformer_hyperparms:
            del transformer_hyperparms["use_enhanced_decoder"]
            
        # Create the transformer without an external encoder
        transformer = Transformer(
            use_enhanced_decoder=use_enhanced_decoder,
            **transformer_hyperparms
        )
        
        # Fine-tune the CNN encoder
        encoder.fine_tune(True)
        
        # Setup optimizers for separate models
        image_enc_lr = config["optim_params"]["encoder_lr"]
        parms2update = filter(lambda p: p.requires_grad, encoder.parameters())
        image_encoder_optim = Adam(params=parms2update, lr=image_enc_lr)
        gamma = config["optim_params"]["lr_factors"][0]
        image_scheduler = StepLR(image_encoder_optim, step_size=1, gamma=gamma)

        transformer_lr = config["optim_params"]["transformer_lr"]
        parms2update = filter(lambda p: p.requires_grad, transformer.parameters())
        transformer_optim = Adam(params=parms2update, lr=transformer_lr)
        gamma = config["optim_params"]["lr_factors"][1]
        transformer_scheduler = StepLR(transformer_optim, step_size=1, gamma=gamma)
        
        optimizers = [image_encoder_optim, transformer_optim]
        schedulers = [image_scheduler, transformer_scheduler]
        

    else:  # hierarchical_cnn_vit
        # For hierarchical, we provide the encoder to the transformer
        transformer_hyperparms["img_encode_size"] = config["hyperparams"]["hierarchical_encoder"]["encode_size"]
        transformer_hyperparms["max_len"] = max_len - 1
        
        # Make sure we don't have duplicate parameters
        if "use_enhanced_decoder" in transformer_hyperparms:
            del transformer_hyperparms["use_enhanced_decoder"]
            
        # Create transformer with the external encoder
        transformer = Transformer(
            use_enhanced_decoder=use_enhanced_decoder, 
            encoder=encoder,
            **transformer_hyperparms
        )
        
        # Set up separate optimizers for encoder and transformer
        # Don't train backbone initially
        if hasattr(encoder, 'backbone'):
            for param in encoder.backbone.parameters():
                param.requires_grad = False
        
        # Encoder optimizer
        encoder_lr = config["optim_params"]["encoder_lr"]
        encoder_params = list(filter(lambda p: p.requires_grad, encoder.parameters()))
        image_encoder_optim = Adam(params=encoder_params, lr=encoder_lr)
        gamma_enc = config["optim_params"]["lr_factors"][0]
        image_scheduler = StepLR(image_encoder_optim, step_size=1, gamma=gamma_enc)
        
        # Transformer optimizer - exclude encoder parameters
        transformer_lr = config["optim_params"]["transformer_lr"]
        transformer_params = list(filter(lambda p: p.requires_grad, 
                                [p for n, p in transformer.named_parameters() 
                                if not n.startswith('encoder')]))
        transformer_optim = Adam(params=transformer_params, lr=transformer_lr)
        gamma_trf = config["optim_params"]["lr_factors"][1]
        transformer_scheduler = StepLR(transformer_optim, step_size=1, gamma=gamma_trf)
        
        optimizers = [image_encoder_optim, transformer_optim]
        schedulers = [image_scheduler, transformer_scheduler]
    
    # Load pretrained embeddings
    print("loading pretrained glove embeddings...")
    weights = vocab.vectors
    transformer.decoder.cptn_emb.from_pretrained(weights,
                                               freeze=True,
                                               padding_idx=pad_id)
    list(transformer.decoder.cptn_emb.parameters())[0].requires_grad = False
    
    # Extract training parameters
    # Make a copy of the train_parms to avoid modifying the original
    train_params = dict(config["train_parms"])
    # Extract the wandb parameters
    use_wandb = train_params.pop("use_wandb", False) if "use_wandb" in train_params else False
    project_name = train_params.pop("project_name", "image-captioning") if "project_name" in train_params else "image-captioning"
    
    # Create the trainer
    train = Trainer(
        optims=optimizers,
        schedulers=schedulers,
        device=device,
        pad_id=pad_id,
        resume=resume,
        checkpoints_path=config["pathes"]["checkpoint"],
        use_wandb=use_wandb,
        project_name=project_name,
        **train_params
    )
    
    # Different training approach based on encoder type
    if encoder_type == "cnn":
        # Original approach with separate encoder and transformer
        train.run(encoder, transformer, [train_iter, val_iter], SEED)
    else:
        # New approach with single combined model
        train.run_combined(transformer, [train_iter, val_iter], SEED)

    print("Training complete!")