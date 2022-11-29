
# Ref: https://pytorch.org/vision/0.8/models.html
# Ref: https://pytorch.org/vision/stable/models.html
# Ref: https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py

# Use ImageNet pretrained model
# alexnet = models.alexnet(pretrained=True)
# alexnet.fc = nn.Linear(512, num_classes=17)

import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.models import alexnet
        
def AlexNet_pretrained(dropout=0.5):
    
    model = alexnet(pretrained=True)
    model.classifier = torch.nn.Sequential(*(list(model.classifier.children())[:-1]), nn.Linear(4096, 17))
    return (model)

# import torch
# import torch.nn as nn
# import models.AlexNet_pretrained as AlexNet_pretrained
# AlexNet_1 = AlexNet_pretrained.AlexNet_pretrained()
# print (AlexNet_1)
# print (AlexNet_1.classifier)
# print (list(AlexNet_1.classifier.children())[:-1])
# print (torch.nn.Sequential(*(list(AlexNet_1.classifier.children())[:-1])))
# print (torch.nn.Sequential(*(list(AlexNet_1.classifier.children())[:-1]), nn.Linear(4096, 17)))
        