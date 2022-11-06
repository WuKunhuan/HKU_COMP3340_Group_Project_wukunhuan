
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    
    def __init__(self, num_classes): 
        
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.relu1 = nn.ReLU(inplace = True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.relu2 = nn.ReLU(inplace = True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.ReLU(inplace = True)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4 = nn.ReLU(inplace = True)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5 = nn.ReLU(inplace = True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.dropout1 = nn.Dropout(p=0.5)
        self.fully_connected1 = nn.Linear(9216, 4096)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fully_connected2 = nn.Linear(4096, 4096)
        self.fully_connected3 = nn.Linear(4096, num_classes)
        
    def forward(self, x):

        if (x.size()[1:4] != torch.Size([3, 224, 224])): 
            raise Exception(f"Error: AlexNet input dimension expected torch.Size([3, 224, 224]), but {x.size()[1:4]} received instead. ")
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)
        x = torch.flatten(x, 1, 3)
        x = self.fully_connected1(x)
        x = self.dropout1(x)
        x = self.fully_connected2(x)
        x = self.dropout2(x)
        x = self.fully_connected3(x)
        
        return x