
# Ref: https://pytorch.org/vision/0.8/models.html
# Ref: https://pytorch.org/vision/stable/models.html
# Ref: https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py

# Note: Remember to install the latest version of torchvision

import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.models import swin_t
        

def SwinTransformer_pretrained():
    model = swin_t(pretrained=True)
    model.head = nn.Linear(768, 17)
    return model
 
        