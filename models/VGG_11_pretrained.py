
# Ref: https://pytorch.org/vision/0.8/models.html
# Ref: https://pytorch.org/vision/stable/models.html
# Ref: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py


import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.models import vgg11
        
def VGG_11_pretrained(dropout=0.5):
    
    model = vgg11(pretrained=True)
    model.classifier = nn.Sequential(*(list(model.classifier.children())[:-1]), nn.Linear(4096, 17))
    return (model)

        