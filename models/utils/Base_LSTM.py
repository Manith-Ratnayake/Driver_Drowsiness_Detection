import torch
import torch.nn as nn
from Base_Encoder import CNNEncoder


class CNNLSTM(nn.Module):

    def __init__(
        self,
        backbone_model,
        encoder_output_dim,
        encoder_dropout,
        lstm_hidden_dim,
        lstm_num_layers=1,
        lstm_dropout=0.0
    ):
        super().__init__()

        self.encoder = CNNEncoder(
            backbone_model=backbone_model,
            encoder_output_dim=encoder_output_dim,
            encoder_dropout=encoder_dropout
        )

        self.lstm = nn.LSTM(
            input_size=encoder_output_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0
        )

        self.classifier = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape  # [B, T, C, H, W]

        x = x.view(B * T, C, H, W)
        features = self.encoder(x)
        features = features.view(B, T, -1)

        lstm_out, _ = self.lstm(features)  # [B, T, lstm_hidden_dim]

        last_timestep = lstm_out[:, -1, :]
        logits = self.classifier(last_timestep)

        return logits.squeeze(1)
