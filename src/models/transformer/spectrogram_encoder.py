import torch
import torch.nn as nn
import torch.nn.functional as F

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h, dtype=dtype), torch.arange(w, dtype=dtype), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"

    omega = torch.arange(dim // 4, dtype=dtype) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)  # shape (h*w, dim)

    # Reshape to (h, w, dim)
    pe = pe.view(h, w, dim)
    return pe.type(dtype)

class TransformerEncoder(nn.Module):
    def __init__(self,
                 model_dim,          # Dimension of the model (embedding size)
                 num_heads,          # Number of attention heads in multi-head attention
                 num_layers,         # Number of encoder layers
                 ff_dim,             # Dimension of the feed-forward layer
                 patch_size,         # Size of the patches to split spectrograms into
                 dropout=0.1,        # Dropout rate
                 freq_max_len=129,
                 time_max_len=71,
                 max_len=128):        # Maximum sequence length (for positional encoding)
        super(TransformerEncoder, self).__init__()
        
        self.model_dim = model_dim
        self.patch_size = patch_size
        
        # Calculate the number of patches
        self.num_patches = (freq_max_len // patch_size) * (time_max_len // patch_size)
        
        # Positional Encoding
        # self.positional_encoding = nn.Embedding(self.num_patches, model_dim)
        # self.positional_encoding = posemb_sincos_2d(129//patch_size, 71//patch_size, model_dim)
        self.pos_embedding = posemb_sincos_2d(
            h = time_max_len // patch_size,
            w = freq_max_len // patch_size,
            dim = model_dim
        )
        
        # Patch Embedding Layer
        self.patch_embedding = nn.Conv2d(1, model_dim, kernel_size=patch_size, stride=patch_size)
        
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
        x = x.unsqueeze(dim=1)

        # Patchify the spectrogram (x is of shape (batch_size, seq_len, freq_len))
        x = self.patch_embedding(x)  # Output shape: (batch_size, model_dim, num_patches)
        
        x = x.permute(0, 2, 3, 1)
        batch_size, t_dim, f_dim, model_dim = x.shape

        print(x[0, 0, 0, :20])
        # add positional embedding
        x = x + self.pos_embedding[:t_dim, :f_dim, :].unsqueeze(0).repeat(batch_size, 1, 1, 1).to(x.device)

        print(x[0, 0, 0, :20])
        x = x.reshape(batch_size, -1, model_dim)

        # Reshape to (num_patches, batch_size, model_dim) as required by Transformer
        x = x.permute(1, 0, 2)  # (num_patches, batch_size, model_dim)

        # Pass through the Transformer Encoder
        x = self.transformer_encoder(x)
        
        # Pooling (taking the mean over the sequence dimension)
        out = x.mean(dim=0)  # (batch_size, model_dim)
        
        return out
