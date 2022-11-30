
# Ref: https://pytorch.org/vision/0.8/models.html
# Ref: https://pytorch.org/vision/stable/models.html
# Ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

# Use ImageNet pretrained model
# resnet18 = models.resnet18(pretrained=True)
# resnet18.fc = nn.Linear(512, num_classes=17)

import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.models import resnet18
        
def ResNet_18_pretrained():
    
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(512, 17)
    return model
 
        