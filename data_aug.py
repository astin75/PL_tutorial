import json

import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import numpy as np



class classificationDataLoader(Dataset):
    def __init__(self, data_path, transform=False):
        super().__init__()
        self.inputShape = (224, 224)
        self.imgPath = data_path
        if transform:
            self.transforms = transform_options()
        else:
            self.transformsFlag = False
            #transforms.compose :Composes several transforms together.
            #ToTensor() Convert a PIL Image or numpy.ndarray to tensor.
            #normalize(tensor, mean, std[, inplace]) Normalize a float tensor image with mean and standard deviation.

            self.transforms = transforms.Compose([
                transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.datset_dict = self.json_read(data_path)

    def json_read(self, path):
        with open(path, "r") as f:
            json_file = json.load(f)
        return json_file['dataset']

    def __len__(self):
        return len(self.datset_dict)

    def __getitem__(self, idx):
        imgFile = self.datset_dict[idx]
        imgPath = imgFile['path']
        yLabel = imgFile['class_num']

        img = cv2.imread(imgPath)
        img = cv2.resize(img, (self.inputShape[0], self.inputShape[1]))

        x = self.transforms(image=img)['image']
        y = torch.tensor(yLabel)

        return {
            'image': x,
            'label': y
        }

def cpu_imshow(image, gt, predict=False,one_hot=False):
    for n, i in enumerate(image):
        img = i.numpy()
        print(img.shape)
        gt = gt.numpy()[0]
        img = np.transpose(img, (1, 2, 0)) #H,W,C
        if one_hot==False:
            print("GT:", gt)
        else:
            if predict !=False:
                predict = predict.numpy()[0]
                print("GT : {0}, Predict : {1}".format(one_hot[str(gt)], one_hot[str(predict)]))
            else:
                print("GT : {0}".format(one_hot[str(gt)]))

        cv2.imshow("ee", img)
        cv2.waitKey(0)



def transform_options():
    transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
        ToTensorV2()
    ])
    return transform

if __name__ == "__main__":
    dataset = classificationDataLoader("train.json", transform=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataiter = iter(dataloader)
    with open("train.json", "r") as f:
        json_file = json.load(f)

    for epoch in range(500):
        for i, data in enumerate(dataloader, 0):

            cpu_imshow(data['image'], data['label'], one_hot=json_file['one_hot'])