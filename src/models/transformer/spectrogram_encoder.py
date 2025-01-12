import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self,
                 model_dim,          # Dimension of the model (embedding size)
                 num_heads,          # Number of attention heads in multi-head attention
                 num_layers,         # Number of encoder layers
                 ff_dim,             # Dimension of the feed-forward layer
                 patch_size,         # Size of the patches to split spectrograms into
                 dropout=0.1,        # Dropout rate
                 max_len=128):        # Maximum sequence length (for positional encoding)
        super(TransformerEncoder, self).__init__()
        
        self.model_dim = model_dim
        self.patch_size = patch_size
        
        # Calculate the number of patches
        self.num_patches = (max_len // (patch_size - 2)) ** 2  # assuming input_dim is divisible by patch_size
        
        # Positional Encoding
        self.positional_encoding = nn.Embedding(self.num_patches, model_dim)
        
        # Patch Embedding Layer
        self.patch_embedding = nn.Conv2d(1, model_dim, kernel_size=patch_size, stride=patch_size - 2)
        
        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=ff_dim, 
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor):
        """
        x: Input tensor of shape (batch_size, time-domain, frequency-domain)
        """
        batch_size, _, _ = x.shape
        x = x.unsqueeze(dim=1)

        # Patchify the spectrogram (x is of shape (batch_size, seq_len, freq_len))
        x = self.patch_embedding(x)  # Output shape: (batch_size, model_dim, num_patches)
        
        x = x.reshape(batch_size, self.model_dim, -1)
        x = x.permute(0, 2, 1)
        
        num_patches = x.shape[1]

        # Add positional encoding
        position = torch.arange(0, num_patches).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        x = x + self.positional_encoding(position)  # Add positional encoding
        
        # Reshape to (num_patches, batch_size, model_dim) as required by Transformer
        x = x.permute(1, 0, 2)  # (num_patches, batch_size, model_dim)
        
        # Pass through the Transformer Encoder
        x = self.transformer_encoder(x)
        
        # Pooling (taking the mean over the sequence dimension)
        out = x.mean(dim=0)  # (batch_size, model_dim)
        
        return out
