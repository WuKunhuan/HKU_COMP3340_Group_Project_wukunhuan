
import torch
import torch.nn as nn

class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))
    
class Inception_block(nn.Module):
    
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()
        self.branch1 = Conv_block(in_channels, out_1x1, kernel_size=(1, 1))
        self.branch2 = nn.Sequential(
        Conv_block(in_channels, red_3x3, kernel_size=(1, 1)),
        Conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)))
        self.branch3 = nn.Sequential(
        Conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
        Conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)))
        self.branch4 = nn.Sequential(
        nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        Conv_block(in_channels, out_1x1pool, kernel_size=(1, 1)))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
    
class Inception_auxiliary(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(Inception_auxiliary, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size = 5, stride = 3)
        self.conv = Conv_block(in_channels, 128, kernel_size = 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, out_channels)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
class Inception_V1(nn.Module):
    
    def __init__(self, num_classes, aux_logits = True, init_weights=False): 
        super(Inception_V1, self).__init__()
        self.aux_logits = aux_logits
        self.conv_1 = Conv_block(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(1, 1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv_2 = Conv_block(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)) 
        self.conv_3 = Conv_block(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = Inception_block(832, 384, 192, 384, 48, 128, 128)
        self.avg_pool_1 = nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1), padding=(0, 0))
        self.dropout = nn.Dropout(p=0.4)
        self.fully_connected_1 = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim = 1)
        if self.aux_logits:
            self.auxiliary_1 = Inception_auxiliary(512, num_classes)
            self.auxiliary_2 = Inception_auxiliary(528, num_classes)
        
    def forward(self, x):
        
        if (x.size()[1:4] != torch.Size([3, 224, 224])): 
            raise Exception(f"Error: Inception_V1 input dimension expected torch.Size([3, 224, 224]), but {x.size()[-3:0:1]} received instead. ")
            
        aux1 = None; aux2 = None
        x = self.conv_1(x)
        x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.max_pool_2(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.max_pool_3(x)
        x = self.inception_4a(x)
        if self.training and self.aux_logits: 
            aux1 = self.auxiliary_1(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        if self.training and self.aux_logits: 
            aux2 = self.auxiliary_2(x)
        x = self.inception_4e(x)
        x = self.max_pool_4(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avg_pool_1(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fully_connected_1(x)
        x = self.softmax(x)
        # print ("Inception_V1 network forward propagation done. ")
                
        if self.aux_logits and self.training: 
            return aux1, aux2, x
        else: return x