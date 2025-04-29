from copy import deepcopy
from typing import Tuple

import torch
from torch import nn, Tensor

from .encoder_layers import EncoderLayer
from .decoder_layers import DecoderLayer, EnhancedDecoderLayer
from .pe import PositionalEncoding

class Encoder(nn.Module):
    """Encoder class that supports both sequence-first and batch-first formats"""
    def __init__(self, layer: EncoderLayer, num_layers: int):
        super().__init__()
        # Make copies of the encoder layer
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)])
        # Store batch_first setting from the layer
        self.batch_first = layer.batch_first

    def forward(self, x: Tensor) -> Tensor:
        # Check input format against expected format
        is_batch_first = (x.dim() == 3 and x.size(0) > x.size(1))
        
        # If input doesn't match expected format, transpose
        if is_batch_first != self.batch_first:
            x = x.transpose(0, 1)
            
        # Process through layers
        for layer in self.layers:
            x = layer(x)
            
        # Return in the original format if we transposed
        if is_batch_first != self.batch_first:
            x = x.transpose(0, 1)
            
        return x

class Decoder(nn.Module):
    """Decoder class that can work with both original and enhanced versions"""
    def __init__(self,
                 layer,
                 vocab_size: int,
                 d_model: int,
                 num_layers: int,
                 max_len: int,
                 dropout: float,
                 pad_id: int,
                 enhanced: bool = False):
        super().__init__()

        self.pad_id = pad_id
        self.enhanced = enhanced

        # Embedding layer + pos encoding
        self.cptn_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEncoding(d_model, max_len)

        # Make copies of the decoder layer
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)])

        self.dropout = nn.Dropout(p=dropout)

    def get_attn_subsequent_mask(self, sz: int) -> Tensor:
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        """
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, tgt_cptn: Tensor, src_img: Tensor) -> Tuple[Tensor, Tensor]:
        """
        param:
        tgt_cptn:   Captions (Transformer target sequence)
                    Tensor [batch_size, max_len-1]

        src_img:    Encoded images (Transformer source sequence)
                    Tensor [encode_size^2, batch_size, image_embed_dim]
                    or [batch_size, encode_size^2, embed_dim] for hierarchical

        outputs:
        output:     Decoder output
                    Tensor [max_len, batch_size, model_embed_dim]

        attn_all:   Attension weights
                    Tensor [layer_num, batch_size, head_num, max_len-1, encode_size^2]
        """
        # Handle different input formats
        # If src_img has shape [batch_size, encode_size^2, embed_dim] (from hierarchical encoder)
        # Transform to [encode_size^2, batch_size, embed_dim] (what the decoder expects)
        if src_img.dim() == 3 and src_img.size(0) == tgt_cptn.size(0):
            # This is in format [batch_size, seq_len, embed_dim]
            src_img = src_img.permute(1, 0, 2)
        
        # Create masks, then pass to decoder
        tgt_pad_mask = (tgt_cptn == self.pad_id)
        tgt_mask = self.get_attn_subsequent_mask(tgt_cptn.size()[1])
        tgt_mask = tgt_mask.to(tgt_cptn.device)

        # Encode captions + pos enc
        # (B, max_len) -> (B, max_len, d_model) -> (max_len, B, d_model)
        tgt_cptn = self.cptn_emb(tgt_cptn)  # type: Tensor
        tgt_cptn = self.dropout(self.pos_emb(tgt_cptn.permute(1, 0, 2)))

        attns_all = []
        for layer in self.layers:
            tgt_cptn, attns = layer(tgt_cptn, src_img, tgt_mask, tgt_pad_mask)
            attns_all.append(attns)
        
        # [layer_num, batch_size, head_num, max_len, encode_size**2]
        attns_all = torch.stack(attns_all)

        return tgt_cptn, attns_all

class Transformer(nn.Module):
    """Transformer that supports both original and enhanced versions with batch_first support"""
    # In transformer.py, modify the Transformer.__init__() method
    def __init__(self,
                vocab_size: int,
                d_model: int,
                img_encode_size: int,
                enc_ff_dim: int,
                dec_ff_dim: int,
                enc_n_layers: int,
                dec_n_layers: int,
                enc_n_heads: int,
                dec_n_heads: int,
                max_len: int,
                dropout: float = 0.1,
                pad_id: int = 0,
                use_enhanced_decoder: bool = False,
                batch_first: bool = False,
                encoder=None):
        super(Transformer, self).__init__()

        
        # If an external encoder is provided, use it (for hierarchical)
        self.encoder = encoder
        self.external_encoder = encoder is not None
        
        # For the original architecture, create encoder layers
        if not self.external_encoder:
            encoder_layer = EncoderLayer(
                img_encode_size=img_encode_size,
                img_embed_dim=d_model,
                feedforward_dim=enc_ff_dim,
                num_heads=enc_n_heads,
                dropout=dropout,
                batch_first=batch_first
            )
            self.encoder = Encoder(layer=encoder_layer, num_layers=enc_n_layers)
        
        # Create decoder layer based on configuration
        if use_enhanced_decoder:
            decoder_layer = EnhancedDecoderLayer(
                d_model=d_model,
                num_heads=dec_n_heads,
                feedforward_dim=dec_ff_dim,
                dropout=dropout
            )
        else:
            decoder_layer = DecoderLayer(
                d_model=d_model,
                num_heads=dec_n_heads,
                feedforward_dim=dec_ff_dim,
                dropout=dropout
            )
        
        self.decoder = Decoder(
            layer=decoder_layer,
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=dec_n_layers,
            max_len=max_len,
            dropout=dropout,
            pad_id=pad_id,
            enhanced=use_enhanced_decoder
        )

        self.predictor = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, images: Tensor, captions: Tensor) -> Tuple[Tensor, Tensor]:
        """
        param:
        image:      source images
                    [batch_size, encode_size^2=196, image_feature_size=512]
                    or [batch_size, 3, h, w] for hierarchical

        captions:   target captions
                    [batch_size, max_len-1=51]

        outputs:
        predictions:    Decoder output
                        Tensor
                        [batch_size, max_len, vocab_size]

        attn_all:       Attention weights
                        Tensor
                        [layer_num, batch_size, head_num, max_len,
                        encode_size^2]
        """
        # For hierarchical encoder, input is raw images
        if self.external_encoder:
            # Hierarchical encoder outputs in batch_first format
            images_encoded = self.encoder(images)  # [B, encode_size^2, embed_dim]
        else:
            # Original architecture
            if images.dim() == 4 and images.size(1) == 3:
                # Raw images: [B, 3, H, W] -> already being handled by encoder
                # Check if we have a suitable encoder for raw images
                if not hasattr(self.encoder, 'handle_raw_images') or not self.encoder.handle_raw_images:
                    raise ValueError("Raw images provided but no suitable encoder available. " +
                                 "Please use hierarchical_cnn_vit encoder_type in config when passing raw images.")
                images_encoded = self.encoder(images)
            elif images.dim() == 3:
                # Handle feature vector inputs based on encoder's expected format
                batch_first = hasattr(self.encoder, 'batch_first') and self.encoder.batch_first
                
                # Check if we need to transpose
                if batch_first and images.size(0) < images.size(1):  # Need seq_len first -> batch first
                    images = images.transpose(0, 1)
                elif not batch_first and images.size(0) > images.size(1):  # Need batch first -> seq_len first
                    images = images.transpose(0, 1)
                    
                # Pass to encoder
                images_encoded = self.encoder(images)
            else:
                raise ValueError(f"Unexpected image tensor shape: {images.shape}")
        
        # Decode and predict
        tgt_cptn, attns = self.decoder(captions, images_encoded)
        predictions = self.predictor(tgt_cptn).permute(1, 0, 2)  # [B, max_len, vocab_size]

        return predictions.contiguous(), attns.contiguous()