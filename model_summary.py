from models import classification
import torch
if __name__ == '__main__':
    model = classification.classification_network(class_num=8).cuda()
    from torchsummary import summary

    summary(model, (3, 224, 224))
    dummy_data = torch.empty(16, 3, 224, 224, dtype = torch.float32).cuda()
    torch.onnx.export(model, dummy_data, "output1.onnx")