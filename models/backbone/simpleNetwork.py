from torch import nn
from ..modules.common import Conv2D_BN
# class simpleNet(nn.Module):
#     def __init__(self):
#         super(simpleNet, self).__init__()

class simpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D_BN(3, nn.ReLU(), 64, 3, 2)
        self.conv2 = Conv2D_BN(64, nn.ReLU(), 128, 3, 2)
        self.conv3 = Conv2D_BN(128, nn.ReLU(), 256, 3, 2)
        self.conv4 = Conv2D_BN(256, nn.ReLU(), 64, 3, 2)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
