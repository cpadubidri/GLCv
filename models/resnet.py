from torchvision.models.resnet import ResNet, BasicBlock
from torch import device, cuda
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import pretrainedmodels
import torch

#sample model
device = device('cuda' if cuda.is_available() else 'cpu')
N_classes = 17036
class ResNetGeolife(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [3, 4, 6, 3], num_classes=N_classes)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)

