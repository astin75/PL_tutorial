import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
import platform
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from models import models
from models import classification
from data_aug import classificationDataLoader
from pl_models import plModel
import warnings

a= torch.cuda.is_available()
b=torch.cuda.current_device()
c=torch.cuda.get_device_name(0)
print(a,b,c)

cfg = {}
cfg['save_freq'] = 5
cfg['save_dir'] = "logs"
cfg['model_name'] = "denseNet"
cfg['epochs'] = 100
cfg['gpus'] = -1
cfg['trainer_options'] = {"check_val_every_n_epoch": 1}

# engine = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if engine == torch.device('cpu'):
#     warnings.warn('Cannot use CUDA context. Train might be slower!')

dataset = classificationDataLoader("train.json", transform=True)
data_module = DataLoader(dataset,
                         batch_size=16,
                         num_workers=4,
                         shuffle=True,
                         pin_memory=4 > 0,
                        persistent_workers=4 > 0
                         )

engine = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if engine == torch.device('cpu'):
    warnings.warn('Cannot use CUDA context. Train might be slower!')
model = classification.classification_network(class_num=8).cuda()
from torchsummary import summary
summary(model, (3, 224, 224))
# dummy_data = torch.empty(16, 3, 224, 224, dtype = torch.float32).cuda()
# torch.onnx.export(model, dummy_data, "output.onnx")
# aa = next(model.parameters()).is_cuda
# print(aa,"e")
#model = models.resnext50(dims=8)
train_module = plModel(model)

callbacks = [
    LearningRateMonitor(logging_interval='step'),
    ModelCheckpoint(monitor='val_top1', save_last=True,
                    every_n_epochs=cfg['save_freq'])
]

trainer = pl.Trainer(
    max_epochs=cfg['epochs'],
    logger=TensorBoardLogger(cfg['save_dir'],
                             cfg['model_name']),
    #devices=1,
    gpus=cfg['gpus'],
    #accelerator =cfg['gpus'],
    accelerator='ddp' if platform.system() != 'Windows' else None,
    plugins=DDPPlugin(
        find_unused_parameters=False) if platform.system() != 'Windows' else None,
    callbacks=callbacks,
    **cfg['trainer_options'])
if __name__ == '__main__':
    trainer.fit(train_module, data_module)
    #tensorboard --logdir logs

