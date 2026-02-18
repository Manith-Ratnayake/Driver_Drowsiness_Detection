import torch 
import torch.nn as nn
import torch.functional as F
from torchvision import utils
from torchvision import models
from torchvision.models import MOBILENET_V2_WEIGHTS


class TemporalAttention(nn.Module):
    
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Linear( 3, 1) 


    def forward(self, x):
        attention_weights = self.attention(x)
        attention_weights = torch.softmax(attention_weights, dim=1)

        weighted_attention_weights = x * attention_weights
        output = weighted_attention_weights.sum(dim=1)
        return output, attention_weights


class TemporalConvolutionBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.groupnorm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.relu      = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None 


    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :-self.conv.padding[0]]
        out = self.groupnorm(out)
        out = self.relu(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res
    

class TCN(nn.Module):
    
    def __init__(self, input_dim, num_channels=[256, 256, 256], kernel_size=3):
        super().__init__()

        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i 
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(TemporalConvolutionBlock(in_channels, out_channels, kernel_size, dilation))
        self.network = nn.Sequential(*layers)


    def forward(self, x):
        return self.network(x)





class TCAN(nn.Module):

    def __init__(self, feature_dim=1280, num_classes=1, pretrained=True):
        backbone = models.mobilenet_v2(pretrained=MOBILENET_V2_WEIGHTS)
        self.cnn = backbone.features()
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.tcn = TCN(input_dim=feature_dim)
        self.attention = TemporalAttention(256)
        self.fc   = nn.Linear(256, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)
        features = self.pool(x)
        features = features.view(B, T, -1)
        features = features.permute(0, 2, 1)
        tcn_out  = self.tcn(features)
        tcn_out  = self.permute(0, 2, 1)
        attn_out, attn_weights = self.attn(tcn_out)
        logits = self.fcn(attn_out)
        return logits, attn_weights
    
