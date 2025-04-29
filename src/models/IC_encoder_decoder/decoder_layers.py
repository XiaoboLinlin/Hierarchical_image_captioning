from typing import Tuple
from torch import nn, Tensor
from torch.nn import MultiheadAttention


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int,
                 dropout: float):
        super(DecoderLayer, self).__init__()
        """
        param:
        d_model:    features size.
                    int

        num_heads:  number of heads in the multiheadattention model.
                    int

        dropout:    dropout value
                    float
        """

        self.dec_self_attn = MultiheadAttention(d_model,
                                                num_heads,
                                                dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model,
                                                 num_heads,
                                                 dropout=dropout)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.multihead_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.multihead_dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                                nn.Linear(feedforward_dim, d_model))

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, dec_inputs: Tensor, enc_outputs: Tensor,
                tgt_mask: Tensor,
                tgt_pad_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        param:
        dec_inputs:     Captions to decode
                        Tensor
                        [max_len, batch_size, embed_dim]

        enc_outputs:    Encoded image to decode
                        Tensor
                        [encode_size^2=196, batch_size, embed_dim]

        tgt_mask:       Mask to ensure that decoder doesn't look at future
                        tokens from a given subsequence
                        [max_len , max_len]

        tgt_pad_mask:   Mask to ensure that decoder doesn't attend pad tokens
                        [batch_size , max_len]

        outputs:
        output:         Decoder output
                        Tensor
                        [max_len, batch_size, embed_dim]

        attn:           Attension weights
                        Tensor
                        [layer_num, batch_size, head_num, max_len,
                        encode_size^2]
                        To be able to do so, I have changed the code at
                        /.virtualenvs/<env_name>/lib/python3.8/site-packages/torch/nn/functional.py
                        line 4818 and changed
                        `return attn_output, attn_output_weights.sum(dim=1) /
                        num_heads` to be
                        `return attn_output, attn_output_weights`

        """
        # self attention + resedual summation + norm
        output, _ = self.dec_self_attn(dec_inputs,
                                       dec_inputs,
                                       dec_inputs,
                                       attn_mask=tgt_mask,
                                       key_padding_mask=tgt_pad_mask)
        output = dec_inputs + self.self_attn_dropout(output)
        output = self.self_attn_norm(output)  # type: Tensor

        # # self attention + residual + norm + FF
        # output2, attns = self.multihead_attn(output, enc_outputs, enc_outputs)
        output2, attns = self.multihead_attn(output, enc_outputs, enc_outputs, need_weights=True, average_attn_weights=False)
        output = output + self.multihead_dropout(output2)
        output = self.multihead_norm(output)

        output2 = self.ff(output)  # type: Tensor
        output = self.ff_norm(output + self.ff_dropout(output2))

        return output, attns


class EnhancedDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float):
        super(EnhancedDecoderLayer, self).__init__()
        
        # Self-attention for decoder inputs
        self.dec_self_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)
        
        # Cross-attention between decoder and encoder outputs
        self.cross_attn = MultiheadAttention(d_model, num_heads, dropout=dropout)
        
        # Layer normalization and dropout
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.cross_attn_dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),  # Using GELU instead of ReLU
            nn.Dropout(p=dropout),
            nn.Linear(feedforward_dim, d_model)
        )
        
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)
        
    def forward(self, dec_inputs: Tensor, enc_outputs: Tensor,
                tgt_mask: Tensor, tgt_pad_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        params:
        dec_inputs:     Captions to decode
                        Tensor [max_len, batch_size, embed_dim]
                        
        enc_outputs:    Encoded image features
                        Tensor [encode_size^2, batch_size, embed_dim]
                        
        tgt_mask:       Mask to ensure decoder doesn't look at future tokens
                        Tensor [max_len, max_len]
                        
        tgt_pad_mask:   Mask to ensure decoder doesn't attend pad tokens
                        Tensor [batch_size, max_len]
                        
        returns:
        output:         Decoder output
                        Tensor [max_len, batch_size, embed_dim]
                        
        attn:           Attention weights from cross-attention
                        Tensor [batch_size, head_num, max_len, encode_size^2]
        """
        # Self-attention block
        residual = dec_inputs
        x, _ = self.dec_self_attn(
            dec_inputs, dec_inputs, dec_inputs,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_pad_mask
        )
        x = residual + self.self_attn_dropout(x)
        x = self.self_attn_norm(x)
        
        # Cross-attention block
        residual = x
        x, cross_attn_weights = self.cross_attn(
            query=x,
            key=enc_outputs,
            value=enc_outputs,
            need_weights=True,
            average_attn_weights=False  # Return attention weights per head
        )
        x = residual + self.cross_attn_dropout(x)
        x = self.cross_attn_norm(x)
        
        # Feed-forward block
        residual = x
        x = self.ff(x)
        x = residual + self.ff_dropout(x)
        x = self.ff_norm(x)
        
        return x, cross_attn_weights