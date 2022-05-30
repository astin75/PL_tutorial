import torch.nn as nn
from models.backbone.simpleNetwork import simpleNet
from models.backbone.denseNetwork import DenseNet
from models.modules.common import Conv2D_BN

class classification_network(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num
        self.backbone = DenseNet() #simpleNet() 64
        self.classification_head =  nn.Sequential(
            Conv2D_BN(128, nn.ReLU(), 1280, (1, 1)),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, self.class_num, 1)
        )

    def forward(self, input):
        x = self.backbone(input)
        x = self.classification_head(x)
        b, c, _, _ = x.size()
        output = x.view(b, c)

        return output

