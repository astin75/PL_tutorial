from torch import nn
from models.modules.common import TransitionLayer, DenseLayer

class DenseStemLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (7, 7), stride=(2, 2), padding=(3, 3))
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.max_pool1(x)
        return x

class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = DenseStemLayer()
        self.dense1 = DenseLayer([32,6,12,24], [6,6,12,24])
        self.transition_layer1 = TransitionLayer(48,12)

        self.dense2 = DenseLayer([12,12,24,48], [12,12,24,48])
        self.transition_layer2 = TransitionLayer(96,24)

        self.dense3 = DenseLayer([24,24,56,104], [24,32,48,64])
        self.transition_layer3 = TransitionLayer(168,16)

        self.dense4 = DenseLayer([16,16,48,80], [16,32,32,48])

    def forward(self, input):
        x = self.stem(input)
        x = self.dense1(x)
        x = self.transition_layer1(x)
        x = self.dense2(x)
        x = self.transition_layer2(x)
        x = self.dense3(x)
        x = self.transition_layer3(x)
        x = self.dense4(x)

        return x

