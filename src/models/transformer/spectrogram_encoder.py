import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, 
                 input_dim,          # Dimension of input features
                 model_dim,          # Dimension of the model (embedding size)
                 num_heads,          # Number of attention heads in multi-head attention
                 num_layers,         # Number of encoder layers
                 ff_dim,             # Dimension of the feed-forward layer
                 dropout=0.1,        # Dropout rate
                 max_len=80):      # Maximum sequence length (for positional encoding)
        super(TransformerEncoder, self).__init__()
        
        self.model_dim = model_dim
        
        # Positional Encoding
        self.positional_encoding = nn.Embedding(max_len, model_dim)
        
        # Input embedding layer
        self.embedding = nn.Linear(input_dim, model_dim)
        
        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=ff_dim, 
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output Layer (optional for final transformation)
        self.fc_out = nn.Linear(model_dim, model_dim)
        
    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, seq_len, input_dim)
        """
        # Adding positional encoding
        batch_size, seq_len, _ = x.shape
        position = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        x = self.embedding(x) + self.positional_encoding(position)
        
        # Reshape to (seq_len, batch_size, model_dim) as required by the transformer
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, model_dim)
        
        # Pass through the transformer encoder
        x = self.transformer_encoder(x)
        
        # You can apply additional processing here (e.g., pooling, classification layer, etc.)
        # Here, we'll simply take the mean of the sequence
        x = x.mean(dim=0)  # (batch_size, model_dim)
        
        # Output layer (optional)
        out = self.fc_out(x)
        
        return out
