import torch
import torch.nn as nn


class CNNEncoder(nn.Module):

  def __init__(self, backbone_model, encoder_output_dim, encoder_dropout, finetune_start_layer=0):
        super().__init__()
        backbone_first_conv_block = backbone_model.features[0][0]
        backbone_model.features[0][0] = nn.Conv2d(
                    in_channels = 1,
                    out_channels = backbone_first_conv_block.out_channels,
                    kernel_size = backbone_first_conv_block.kernel_size,
                    stride = backbone_first_conv_block.stride,
                    padding = backbone_first_conv_block.padding,
                    bias=False)


        with torch.no_grad():
            backbone_model.features[0][0].weight.copy_(backbone_first_conv_block.weight.mean(dim=1, keepdim=True))

        self.features = backbone_model.features

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        backbone_output_dim = backbone_model.classifier[0].in_features

        self.fc = nn.Sequential(
            nn.Linear(backbone_output_dim, encoder_output_dim),
            nn.LayerNorm(encoder_output_dim),
            nn.GELU(),
            nn.Dropout(encoder_dropout),
        )

        for param in self.features.parameters():
             param.requires_grad = False

        for param in self.features[finetune_start_layer:].parameters():
            param.requires_grad = True


  def forward(self, x):
      x = self.features(x)
      x = self.pool(x)
      x = self.fc(x)
      return x