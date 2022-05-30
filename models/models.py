from torchvision import models
from torch import nn


def resnet50(dims=256):
    resnet = models.resnet50(pretrained=True)
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, dims)
    return resnet


def resnext50(dims=256):
    resnet = models.resnext50_32x4d(pretrained=True)
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, dims)
    return resnet



import torch
if __name__ == '__main__':
    model = resnext50(dims=256).cuda()
    from torchsummary import summary

    summary(model, (3, 224, 224))
    dummy_data = torch.empty(16, 3, 224, 224, dtype = torch.float32).cuda()
    torch.onnx.export(model, dummy_data, "output1.onnx")