
# Ref: https://pytorch.org/vision/0.8/models.html
# Ref: https://pytorch.org/vision/stable/models.html
# Ref: https://pytorch.org/vision/stable/models/googlenet.html
# Ref: https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py


import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.models import googlenet
        
def Inception_V1_pretrained():
    
    model = googlenet(pretrained=True)
    model.fc = nn.Linear(1024, 17)
    return model
 
        