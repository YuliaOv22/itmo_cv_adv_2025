import torch
import torch.nn as nn

class YoloV1(nn.Module):
    def __init__(
        self, 
        backbone: nn.Module, 
        split_size=7, 
        num_boxes=2, 
        num_classes=2, 
        backbone_out_features=1280
    ):
        super(YoloV1, self).__init__()
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

        self.backbone = nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool2d((self.S, self.S))  # Приводим к нужному spatial размеру
        )

        self.fcs = self._create_fcs(split_size, num_boxes, num_classes, backbone_out_features)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        x = x.view(-1, self.S, self.S, self.C + 5 * self.B)
        return x

    def _create_fcs(self, split_size, num_boxes, num_classes, backbone_out_features):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Linear(backbone_out_features * S * S, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, S * S * (C + 5 * B)),
        )
    