
# Ref: https://pytorch.org/vision/0.8/models.html
# Ref: https://pytorch.org/vision/stable/models.html
# Ref: https://pytorch.org/vision/main/models/vision_transformer.html
# Ref: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py

# Note: Remember to install the latest version of torchvision

import torch
import torch.nn as nn
from torchvision.io import read_image
from collections import OrderedDict
from torchvision.models import vit_b_16
        

def VisionTransformer_pretrained():
    num_classes = 17
    
    model = vit_b_16(pretrained=True)
    
    heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
    if model.representation_size is None:
        heads_layers["head"] = nn.Linear(model.hidden_dim, num_classes)
    else:
        heads_layers["pre_logits"] = nn.Linear(model.hidden_dim, model.representation_size)
        heads_layers["act"] = nn.Tanh()
        heads_layers["head"] = nn.Linear(model.representation_size, num_classes)

    model.heads = nn.Sequential(heads_layers)
    
    return model
 
        