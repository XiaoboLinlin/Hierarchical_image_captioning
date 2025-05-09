from torch import nn, Tensor
from torch.nn import MultiheadAttention


class CNNFeedForward(nn.Module):
    """
    ## FFN module
    """

    def __init__(self, encode_size: int, embed_dim: int, feedforward_dim: int,
                 dropout: float):
        super(CNNFeedForward, self).__init__()
        """
        param:
        encode_size:        encoded image size.
                            int

        embed_dim:          encoded images features dimension.
                            int

        feedforward_dim:    feedforward network model features dimension.
                            int

        dropout:            dropout value
                            float
        """
        # TODO:
        # Need to be revisited. Not correct!
        # Based on:
        # https://github.com/RoyalSkye/Image-Caption/blob/e528b36b32fdc8175921ce60bb9a2c6cecafebb8/transformer.py#L73-L93
        # Two fc layers can also be described by two cnn with kernel_size=1.
        # https://sebastianraschka.com/faq/docs/fc-to-conv.html#methods-2-convolution-with-1x1-kernels
        self.conv1 = nn.Conv1d(in_channels=embed_dim,
                               out_channels=feedforward_dim,
                               kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feedforward_dim,
                               out_channels=embed_dim,
                               kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        param:
        inputs: Output from multi head attension layere.
                Tensor [encode_size^2, batch_size, embed_dim]

        output: output tensor.
                Tensor [encode_size^2, batch_size, embed_dim]
        """
        # Handle 4D inputs
        original_shape = inputs.shape
        is_4d = inputs.dim() == 4
        
        if is_4d:
            # Reshape to 3D for processing
            seq_len = original_shape[-2]
            batch_size = original_shape[0] * original_shape[1]
            embed_dim = original_shape[-1]
            inputs = inputs.reshape(-1, seq_len, embed_dim).permute(1, 0, 2)
            
        # For Conv1d: input should be [batch_size, channels, seq_len]
        # So we need to convert [seq_len, batch_size, embed_dim] -> [batch_size, embed_dim, seq_len]
        # permute the dimensions to (1, 0, 2) -> [batch_size, seq_len, embed_dim]
        # then permute to (0, 2, 1) -> [batch_size, embed_dim, seq_len]
        inputs_permuted = inputs.permute(1, 2, 0)  # [batch_size, embed_dim, seq_len]
        
        # Process through conv layers
        conv_output = self.conv2(self.relu(self.conv1(inputs_permuted)))  # [batch_size, embed_dim, seq_len]
        
        # Convert back to original format [seq_len, batch_size, embed_dim]
        conv_output = conv_output.permute(2, 0, 1)  # [seq_len, batch_size, embed_dim]
        
        # Apply dropout and residual connection
        output = self.dropout(conv_output)
        output = self.layer_norm(output + inputs)
        
        # Reshape back if needed
        if is_4d:
            output = output.permute(1, 0, 2).reshape(original_shape)
            
        return output


class EncSelfAttension(nn.Module):

    def __init__(self, img_embed_dim: int, num_heads: int, dropout: float):
        super(EncSelfAttension, self).__init__()
        """
        param:
        img_embed_dim:  encoded images features dimension.
                        int

        num_heads:      number of heads in the multiheadattention model.
                        int

        dropout:        dropout value
                        float
        """
        self.img_embed_dim = img_embed_dim
        self.multi_head_attn = MultiheadAttention(embed_dim=img_embed_dim,
                                                  num_heads=num_heads,
                                                  dropout=dropout)
        self.layer_norm = nn.LayerNorm(img_embed_dim)

    def forward(self, enc_inputs: Tensor) -> Tensor:
        """
        param:
        enc_inputs:     Input images to encode
                        Tensor
                        [encode_size^2, batch_size, embed_dim]

        outputs:
        enc_outputs:    Encoded image
                        Tensor
                        [encode_size^2, batch_size, embed_dim]
        """
        # Handle 4D inputs by reshaping
        original_shape = enc_inputs.shape
        
        if enc_inputs.dim() == 4:
            # Reshape to 3D: [*, *, seq_len, embed_dim] -> [seq_len, batch, embed_dim]
            seq_len = original_shape[-2]
            batch_size = original_shape[0] * original_shape[1]
            embed_dim = original_shape[-1]
            
            if embed_dim != self.img_embed_dim:
                raise ValueError(f"Embedding dimension mismatch: got {embed_dim}, expected {self.img_embed_dim}. Shape: {original_shape}")
                
            enc_inputs_reshaped = enc_inputs.reshape(-1, seq_len, embed_dim).permute(1, 0, 2)
        else:
            # Check embedding dimension for 3D tensors
            if enc_inputs.size(-1) != self.img_embed_dim:
                raise ValueError(f"Embedding dimension mismatch: got {enc_inputs.size(-1)}, expected {self.img_embed_dim}. Shape: {original_shape}")
                
            enc_inputs_reshaped = enc_inputs

        # Apply self-attention
        enc_outputs, _ = self.multi_head_attn(enc_inputs_reshaped, enc_inputs_reshaped,
                                          enc_inputs_reshaped)
        
        # Add residual connection and layer norm
        enc_outputs = enc_outputs + enc_inputs_reshaped
        enc_outputs = self.layer_norm(enc_outputs)
        
        # Reshape back to original shape if needed
        if enc_inputs.dim() == 4:
            enc_outputs = enc_outputs.permute(1, 0, 2).reshape(original_shape)
            
        return enc_outputs

class EncoderLayer(nn.Module):
    def __init__(self, img_encode_size: int, img_embed_dim: int,
                 feedforward_dim: int, num_heads: int, dropout: float, batch_first: bool = False):
        super(EncoderLayer, self).__init__()
        
        self.batch_first = batch_first
        self.enc_self_attn = EncSelfAttension(img_embed_dim=img_embed_dim,
                                             num_heads=num_heads,
                                             dropout=dropout)
                                             
        self.cnn_ff = CNNFeedForward(encode_size=img_encode_size,
                                     embed_dim=img_embed_dim,
                                     feedforward_dim=feedforward_dim,
                                     dropout=dropout)

    def forward(self, enc_inputs: Tensor) -> Tensor:
        """
        param:
        enc_inputs:     Input images to encode
                        Tensor
                        [encode_size^2, batch_size, embed_dim]

        outputs:
        enc_outputs:    Encoded image
                        Tensor
                        [encode_size^2, batch_size, embed_dim]
        """
        enc_outputs = self.enc_self_attn(enc_inputs)
        enc_outputs = self.cnn_ff(enc_outputs)
        return enc_outputs

if __name__ == "__main__":
    import torch

    src_img = torch.rand(196, 10, 512)  # B, encode, embed
    m_test = EncoderLayer(196, 512, 512, 8, 0.1)
    valus = m_test(src_img)
    print(valus.size())
