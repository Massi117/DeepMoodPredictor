import torch
import torch.nn as nn

# Define MoodCNNClassifier model
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class CNN3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()

        self.stage1 = ConvBlock3D(in_channels, 32, stride=1)   # 91×101×91
        self.stage2 = ConvBlock3D(32, 64, stride=2)            # ~46×50×46
        self.stage3 = ConvBlock3D(64, 128, stride=2)           # ~23×25×23
        self.stage4 = ConvBlock3D(128, 256, stride=2)          # ~12×13×12

        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.global_pool(x)    # (B, 256, 1, 1, 1)
        x = x.flatten(1)           # (B, 256)

        return self.classifier(x)
