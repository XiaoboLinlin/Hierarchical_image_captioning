from pathlib import Path
from collections import defaultdict
from statistics import mean, pstdev
from tqdm import tqdm
import pickle
import warnings
import argparse

# Ignore warnings about model loading
warnings.filterwarnings("ignore", category=UserWarning, message=".*missing keys.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*unexpected key.*")

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torchvision.transforms import Normalize, Compose
from torch.utils.data import DataLoader, Subset

from dataset.dataloader import HDF5Dataset, collate_padd
from models.encoder_factory import create_encoder
from models.IC_encoder_decoder.transformer import Transformer
from models.IC_encoder_decoder.decoder_layers import EnhancedDecoderLayer
from nlg_metrics import Metrics
from utils.train_utils import seed_everything
from utils.config_loader import load_config, validate_config
from utils.test_utils import parse_arguments
from utils.gpu_cuda_helper import select_device

# Add defaultdict and list to safe globals for torch.load
torch.serialization.add_safe_globals([defaultdict, list])

def get_datasets(dataset_dir: str, pad_id: float, use_val_data: bool = False):
    """Load either test or validation dataset based on the flag"""
    # Setting some paths
    dataset_dir = Path(dataset_dir)
    
    # Decide which dataset to use (test or val)
    if use_val_data:
        images_path = dataset_dir / "val_images.hdf5"
        captions_path = dataset_dir / "val_captions.json"
        lengthes_path = dataset_dir / "val_lengthes.json"
        print("Using validation data for inference")
    else:
        images_path = dataset_dir / "test_images.hdf5"
        captions_path = dataset_dir / "test_captions.json"
        lengthes_path = dataset_dir / "test_lengthes.json"
        print("Using test data for inference")
    
    # images transform
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = Compose([norm])

    dataset = HDF5Dataset(hdf5_path=str(images_path),
                          captions_path=str(captions_path),
                          lengthes_path=str(lengthes_path),
                          pad_id=pad_id,
                          transform=transform)

    return dataset


if __name__ == "__main__":
    # Parse command arguments
    args = parse_arguments()
    dataset_dir = args.dataset_dir
    
    # Device selection
    device = select_device(args.device)
    print(f"Selected device is {device}.\n")

    # Load YAML configuration
    config = load_config(args.config_path)
    validate_config(config)
    
    # Make sure encoder_type is set to hierarchical_cnn_vit
    config["hyperparams"]["encoder_type"] = "hierarchical_cnn_vit"
    
    # Ensure backbone_name is set for the hierarchical encoder
    if "hierarchical_encoder" not in config["hyperparams"]:
        config["hyperparams"]["hierarchical_encoder"] = {}
    
    config["hyperparams"]["hierarchical_encoder"]["backbone_name"] = "efficientnet_b3"

    # Setting paths
    checkpoints_dir = config["pathes"]["checkpoint"]
    checkpoint_name = args.checkpoint_name
    experiment_name = checkpoint_name.split("/")[0]
    
    # Create save directory
    save_dir = Path(args.save_dir) / f"{experiment_name}"
    if args.use_val_data:
        save_dir = save_dir.with_name(f"{save_dir.name}_val")
    elif config.get("debug", {}).get("enabled", False):
        save_dir = save_dir.with_name(f"{save_dir.name}_test_debug")
    save_dir.mkdir(parents=True, exist_ok=True)

    # SEED
    SEED = config["seed"]
    seed_everything(SEED)

    # Load vocab
    print("Loading vocab...")
    vocab: Vocab = torch.load(str(Path(dataset_dir) / "vocab.pth"))
    pad_id = vocab.stoi["<pad>"]
    sos_id = vocab.stoi["<sos>"]
    eos_id = vocab.stoi["<eos>"]
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # --------------- dataloader --------------- #
    print("Loading dataset...")
    test_dataset = get_datasets(dataset_dir, pad_id, args.use_val_data)

    g = torch.Generator()
    g.manual_seed(SEED)
    max_len = config["max_len"]

    # Get debug settings from config
    debug = config.get("debug", {}).get("enabled", False)
    debug_samples = args.max_debug_samples if debug else 0

    if debug:
        print(f"Running in debug mode with max {debug_samples} samples")
        # don't exceed dataset length
        N = min(debug_samples, len(test_dataset))
        test_dataset = Subset(test_dataset, list(range(N)))

    # Get batch size from config
    batch_size = config.get("dataloader_parms", {}).get("batch_size", 8)
    print(f"Using batch size: {batch_size}")

    test_iter = DataLoader(
        test_dataset,
        collate_fn=collate_padd(max_len, pad_id),
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=False
    )
    print("Loading dataset finished.")

    # --------------- Construct models --------------- #
    print("Constructing models...")

    # Load checkpoint first to check architecture
    load_path = str(Path(checkpoints_dir) / checkpoint_name)
    print(f"Loading checkpoint from {load_path}")
    state = torch.load(load_path, map_location=torch.device("cpu"), weights_only=True)

    # We'll use the config file directly since the model_config is not saved in checkpoints anymore
    print("Using model configuration from YAML config file")
    
    # Debug: Print encoder type before creating encoder
    print(f"Encoder type from config: {config['hyperparams'].get('encoder_type', 'not set')}")
    print(f"Available hyperparams keys: {list(config['hyperparams'].keys())}")
    
    # Create encoder with the config
    encoder = create_encoder(config)

    # Prepare transformer hyperparameters
    transformer_hyperparms = config["hyperparams"]["transformer"]
    transformer_hyperparms["vocab_size"] = vocab_size
    transformer_hyperparms["pad_id"] = pad_id

    # Get encoder type from config
    encoder_type = config["hyperparams"].get("encoder_type", "cnn")
    use_enhanced_decoder = config["hyperparams"].get("transformer", {}).get("use_enhanced_decoder", False)

    if encoder_type == "hierarchical_cnn_vit":
        # For hierarchical, we provide the encoder to the transformer
        transformer_hyperparms["img_encode_size"] = config["hyperparams"]["hierarchical_encoder"]["encode_size"]
        transformer_hyperparms["max_len"] = max_len - 1
        
        # Make sure we don't have duplicate parameters
        if "use_enhanced_decoder" in transformer_hyperparms:
            del transformer_hyperparms["use_enhanced_decoder"]
        
        # Create transformer with the external encoder
        model = Transformer(
            use_enhanced_decoder=use_enhanced_decoder, 
            encoder=encoder,
            **transformer_hyperparms
        )
    else:
        raise ValueError(f"Expected encoder_type 'hierarchical_cnn_vit', got '{encoder_type}'")

    # Load model state
    print(f"Loading model weights...")
    try:
        # Check if this is a combined model (one model state) or separate models
        if len(state["models"]) == 1 and state.get("model_type") == "combined":
            model.load_state_dict(state["models"][0], strict=True)
            print("Combined model weights loaded successfully")
        else:
            # Handle loading separate encoder/decoder states if needed
            print("Warning: Checkpoint contains multiple model states, expected a combined model.")
            print("Attempting to load weights anyway...")
            model.load_state_dict(state["models"][0], strict=False)
            print("Model loaded with strict=False")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load with strict=False as fallback")
        model.load_state_dict(state["models"][0], strict=False)
        print("Model loaded with strict=False")

    # Move model to device
    model.to(device).eval()
    print("Model loaded and ready for inference")

    # Beam search parameters
    beam_size = 10

    # Results tracking
    eval_data = defaultdict(
        list, {
            "hypos_text": [],
            "refs_text": [],
            "attns": [],
            "log_prob": [],
            "bleu1": [],
            "bleu2": [],
            "bleu3": [],
            "bleu4": [],
            "gleu": [],
            "meteor": []
        })
    selected_data = defaultdict(
        list, {
            "hypos_text": [],
            "refs_text": [],
            "attns": [],
            "bleu1": [],
            "bleu2": [],
            "bleu3": [],
            "bleu4": [],
            "gleu": [],
            "meteor": []
        })
    
    # Setup metrics
    nlgmetrics = Metrics()
    h = w = config["hyperparams"]["hierarchical_encoder"]["encode_size"]
    
    # Start inference
    pb = tqdm(test_iter, leave=False, total=len(test_iter))
    pb.unit = "step"
    with torch.no_grad():
        for imgs, cptns_all, lens in pb:
            imgs: Tensor  # images [B, 3, 256, 256]
            cptns_all: Tensor  # all 5 captions [B, lm, cn=5]
            lens: Tensor  # lengthes of all captions [B, cn=5]

            batch_size = imgs.size(0)
            
            # Process each image in the batch
            for b in range(batch_size):
                img = imgs[b:b+1]  # [1, 3, 256, 256]
                cptn_all = cptns_all[b:b+1]  # [1, lm, cn=5]
                len_all = lens[b:b+1]  # [1, cn=5]
                
                # Move to device
                img = img.to(device)
                start = torch.full(size=(1, 1),
                                fill_value=sos_id,
                                dtype=torch.long,
                                device=device)
                
                # Initial step
                logits, attns = model(img, start)
                logits: Tensor  # [k=1, 1, vsc]
                attns: Tensor  # [ln, k=1, hn, S=1, is]
                log_prob = F.log_softmax(logits, dim=2)
                log_prob_topk, indxs_topk = log_prob.topk(beam_size, sorted=True)
                # log_prob_topk [1, 1, k]
                # indices_topk [1, 1, k]
                current_preds = torch.cat(
                    [start.expand(beam_size, 1), indxs_topk.view(beam_size, 1)], dim=1)
                # current_preds: [k, S]

                seq_preds = []
                seq_log_probs = []
                seq_attns = []
                
                # Beam search
                k = beam_size
                while current_preds.size(1) <= (max_len - 2) and k > 0 and current_preds.nelement():
                    # Forward pass
                    logits, attns = model(img.expand(k, *img.size()[1:]), current_preds)
                    # logits: [k, S, vsc]
                    # attns: # [ln, k, hn, S, is]

                    # next word prediction
                    log_prob = F.log_softmax(logits[:, -1:, :], dim=-1).squeeze(1)
                    # log_prob: [k, vsc]
                    log_prob = log_prob + log_prob_topk.view(k, 1)
                    # top k probs in log_prob[k, vsc]
                    log_prob_topk, indxs_topk = log_prob.view(-1).topk(k, sorted=True)
                    # indxs_topk are a flat indices, convert them to 2d indices:
                    # i.e the top k in all log_prob: get indices: K, next_word_id
                    prev_seq_k, next_word_id = np.unravel_index(
                        indxs_topk.cpu(), log_prob.size())
                    next_word_id = torch.as_tensor(next_word_id).to(device).view(k, 1)
                    # prev_seq_k [k], next_word_id [k]

                    current_preds = torch.cat(
                        (current_preds[prev_seq_k], next_word_id), dim=1)

                    # find predicted sequences that ends
                    seqs_end = (next_word_id == eos_id).view(-1)
                    if torch.any(seqs_end):
                        seq_preds.extend(seq.tolist()
                                        for seq in current_preds[seqs_end])
                        seq_log_probs.extend(log_prob_topk[seqs_end].tolist())
                        # get last layer, mean across transformer heads
                        attns = attns[-1].mean(dim=1).view(k, -1, h, w)
                        # [k, S, h, w]
                        seq_attns.extend(attns[prev_seq_k][seqs_end].tolist())

                        k -= torch.sum(seqs_end)
                        current_preds = current_preds[~seqs_end]
                        log_prob_topk = log_prob_topk[~seqs_end]

                # Sort predicted captions according to seq_log_probs
                specials = [pad_id, sos_id, eos_id]
                if seq_preds:
                    seq_preds, seq_attns, seq_log_probs = zip(*sorted(
                        zip(seq_preds, seq_attns, seq_log_probs), key=lambda tup: -tup[2]))

                    text_preds = [[vocab.itos[s] for s in seq if s not in specials]
                                for seq in seq_preds]
                    text_refs = [[vocab.itos[r] for r in ref if r not in specials]
                                for ref in cptn_all.squeeze(0).permute(1, 0)]

                    # calculate scores for each prediction
                    scores = defaultdict(list)
                    for text_pred in text_preds:
                        for k, v in nlgmetrics.calculate([text_refs], [text_pred]).items():
                            scores[k].append(v)

                    # save all eval data
                    eval_data["hypos_text"].append(text_preds)
                    eval_data["refs_text"].append(text_refs)
                    eval_data["attns"].append(list(seq_attns))
                    eval_data["log_prob"].append(list(seq_log_probs))
                    for k, v_list in scores.items():
                        eval_data[k].append(v_list)

                    # save data for the prediction with the highest log_prob
                    selected_data["hypos_text"].append(text_preds[0])
                    selected_data["refs_text"].append(text_refs)
                    selected_data["attns"].append(list(seq_attns)[0])
                    selected_data["bleu1"].append(scores["bleu1"][0])
                    selected_data["bleu2"].append(scores["bleu2"][0])
                    selected_data["bleu3"].append(scores["bleu3"][0])
                    selected_data["bleu4"].append(scores["bleu4"][0])
                    selected_data["gleu"].append(scores["gleu"][0])
                    selected_data["meteor"].append(scores["meteor"][0])

            # tracking data on progress bar (average of current batch)
            if selected_data["bleu4"]:
                pb.set_description(
                    f'bleu4: Current: {selected_data["bleu4"][-batch_size:][-1]:.4f}, Max: {max(selected_data["bleu4"]):.4f}, Min: {min(selected_data["bleu4"]):.4f}, Mean: {mean(selected_data["bleu4"]):.4f} \u00B1 {pstdev(selected_data["bleu4"]):.2f}'  # noqa: E501
                )

    pb.close()

    # save results
    print("\nSaving data...")
    pd.DataFrame(data=eval_data).to_pickle(str(save_dir / "all.pickle"))
    pd.DataFrame(data=selected_data).to_pickle(
        str(save_dir / "selected.pickle"))
    
    # Save metrics to CSV for easier analysis
    metrics_df = pd.DataFrame({
        "bleu1": selected_data["bleu1"],
        "bleu2": selected_data["bleu2"],
        "bleu3": selected_data["bleu3"],
        "bleu4": selected_data["bleu4"],
        "gleu": selected_data["gleu"],
        "meteor": selected_data["meteor"]
    })
    metrics_df.to_csv(str(save_dir / "metrics.csv"), index=False)
    
    # Save example predictions
    examples_df = pd.DataFrame({
        "prediction": [" ".join(hypo) for hypo in selected_data["hypos_text"][:20]],
        "reference": [" ".join(ref[0]) for ref in selected_data["refs_text"][:20]],
        "bleu4": selected_data["bleu4"][:20]
    })
    examples_df.to_csv(str(save_dir / "examples.csv"), index=False)

    # Print metrics summary
    print("\nMetrics Summary:")
    print("-" * 50)
    for metric in ["bleu1", "bleu2", "bleu3", "bleu4", "gleu", "meteor"]:
        values = selected_data[metric]
        print(f"{metric.upper():<10}: Mean: {mean(values):.4f} ±{pstdev(values):.4f}, Min: {min(values):.4f}, Max: {max(values):.4f}")
    print("-" * 50)

    print("Done.")