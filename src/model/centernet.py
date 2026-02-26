import torch
import torch.nn as nn
import torchvision.models as models


resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)


class CenterNet(nn.Module):

    def __init__(self, num_classes=1):
        super(CenterNet, self).__init__()

        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )

        # 3 deconv layers: stride 32 -> 16 -> 8 -> 4
        channels = [512, 256, 128, 64]
        deconv_layers = []

        for i in range(3):
            deconv_layers.extend([
                nn.ConvTranspose2d(channels[i], channels[i + 1], 4, stride=2, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ])
        
        self.deconv = nn.Sequential(*deconv_layers)

        self.heatmap_head = nn.Conv2d(64, num_classes, 1)
        self.offset_head = nn.Conv2d(64, 2, 1)
        self.size_head = nn.Conv2d(64, 2, 1)

        nn.init.constant_(self.heatmap_head.bias, -2.19)

    def forward(self, x):
        h, w = x.shape[2] // 4, x.shape[3] // 4

        features = self.backbone(x)
        features = self.deconv(features)
        features = features[:, :, :h, :w]

        heatmap = torch.sigmoid(self.heatmap_head(features)).squeeze(1)
        offset = self.offset_head(features)
        size = self.size_head(features)

        return heatmap, offset, size