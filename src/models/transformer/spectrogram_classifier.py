import torch
from torch import nn
from src.models.transformer.spectrogram_encoder import TransformerEncoder


class Classifier(nn.Module):
    def __init__(
        self,
        encoder: TransformerEncoder,
        n_classes: int = 94,
        hidden_dim: int = 256
    ):
        super(Classifier, self).__init__()
        
        self.encoder = encoder

        self.readout = nn.Sequential(
            nn.Linear(self.encoder.model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x):
        spec_feat, stat_feat = x

        spec_feat = spec_feat.permute(0, 2, 1)

        encoded = self.encoder(spec_feat)

        out = self.readout(encoded)

        return out