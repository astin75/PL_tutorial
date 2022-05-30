import math
import torch.nn as nn
import torch
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

class Conv2D_BN(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride=1, padding='same', groups=1, dilation=1, bias=False):
        super().__init__()

        self.activation = activation
        in_channels = math.floor(in_channels)
        out_channels = math.floor(out_channels)
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.batchNorm_layer = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        output = self.conv_layer(input)
        output = self.batchNorm_layer(output)
        if self.activation != None:
            output = self.activation(output)
        return output


class DenseBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        in_channel = math.floor(in_channel)
        out_channel = math.floor(out_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, (1, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(out_channel, out_channel, (3, 3), stride=(1, 1), padding=(1, 1))
        self.batch1 = nn.BatchNorm2d(out_channel)
        self.activation = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channel, out_channel, (1, 1), stride=(1, 1))

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.batch1(x)
        x = self.activation(x)
        x = self.conv3(x)
        return x

class DenseLayer(nn.Module):
    def __init__(self, in_layers_list, out_layers_list):
        super().__init__()
        self.layers = []
        self.layer1 = DenseBlock(in_channel=math.floor(in_layers_list[0]),
                                 out_channel=math.floor(out_layers_list[0])).cuda()
        self.layer2 = DenseBlock(in_channel=math.floor(in_layers_list[1]),
                                 out_channel=math.floor(out_layers_list[1])).cuda()
        self.layer3 = DenseBlock(in_channel=math.floor(in_layers_list[2]),
                                 out_channel=math.floor(out_layers_list[2])).cuda()
        self.layer4 = DenseBlock(in_channel=math.floor(in_layers_list[3]),
                                 out_channel=math.floor(out_layers_list[3])).cuda()

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        xc2 = torch.cat([x2, x1], 1)
        x3 = self.layer3(xc2)
        xc3 = torch.cat([x3, xc2], 1)
        x4 = self.layer4(xc3)
        xc4 = torch.cat([x4, xc3], 1)

        return xc4


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        in_channels = math.floor(in_channels)
        out_channels = math.floor(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1,1))
        self.avg_pool1 = nn.AvgPool2d(kernel_size=(2,2), stride=2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.avg_pool1(x)
        return x

