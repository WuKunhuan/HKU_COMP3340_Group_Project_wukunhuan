
# Ref: https://pytorch.org/vision/0.8/models.html
# Ref: https://pytorch.org/vision/stable/models.html
# Ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py


import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.models import resnet50
        
def ResNet_50_pretrained():
    
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(512, 17)
    return model
 
        