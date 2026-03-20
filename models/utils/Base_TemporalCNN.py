import torch
import torch.nn as nn
from Base_Encoder import CNNEncoder

class CausalTemporalConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation, temporal_dropout, num_groups=8, kernel_size=3):
        super().__init__()

        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(temporal_dropout)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):

        residual = self.residual(x)

        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv3(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.norm3(out)
        out = self.activation(out)
        out = self.dropout(out)

        return self.activation(out + residual)



class CNNTemporalConv(nn.Module):

    def __init__(self, backbone_model, encoder_output_dim, encoder_dropout, temporal_dropout, tcn_hidden_dim):
        super().__init__()

        self.encoder = CNNEncoder(
            backbone_model=backbone_model,
            encoder_output_dim=encoder_output_dim,
            encoder_dropout=encoder_dropout
        )

        tcn_channels = tcn_hidden_dim
        num_tcn_blocks = len(tcn_channels)

        if num_tcn_blocks < 1:
            raise ValueError("num_tcn_blocks must be at least 1")

        tcn_dilations = [2 ** i for i in range(num_tcn_blocks)]

        blocks = []
        in_channels = encoder_output_dim

        for out_channels, dilation in zip(tcn_channels, tcn_dilations):

            blocks.append(
                CausalTemporalConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dilation=dilation,
                    temporal_dropout=temporal_dropout
                )
            )

            in_channels = out_channels

        self.tcn = nn.Sequential(*blocks)
        self.classifier = nn.Linear(in_channels, 1)


    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        features = self.encoder(x)
        features = features.view(B, T, -1)
        features = features.transpose(1, 2)

        tcn_out = self.tcn(features)
        last_timestep = tcn_out[:, :, -1]
        logits = self.classifier(last_timestep)

        return logits.squeeze(1)