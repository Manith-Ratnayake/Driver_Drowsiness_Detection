import torch
import torch.nn as nn
from models.utils.architectures.Base_Encoder import CNNEncoder


class CNNTemporalTransformer(nn.Module):

    def __init__(self, backbone_model, encoder_output_dim, encoder_dropout,
                 temporal_dropout, num_heads=4, num_layers=2, max_len=32):

        super().__init__()

        self.encoder = CNNEncoder(
            backbone_model=backbone_model,
            encoder_output_dim=encoder_output_dim,
            encoder_dropout=encoder_dropout
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, encoder_output_dim))

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=encoder_output_dim,
            nhead=num_heads,
            dropout=temporal_dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(encoder_output_dim, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        features = self.encoder(x)
        features = features.view(B, T, -1)

        features = features + self.pos_embedding[:, :T, :]
        transformed = self.transformer(features)
        last_timestep = transformed[:, -1, :]
        logits = self.classifier(last_timestep)

        return logits.squeeze(1)
