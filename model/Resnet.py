import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear, Softmax
from torchvision.transforms import Resize, InterpolationMode
import torch.nn.functional as F

class Basic_Block(Module):
    def __init__(self, Proc_Channel, DownSample):
        super(Basic_Block, self).__init__()
        Proc_Channel= int(Proc_Channel)
        stride = 1
        in_channels = int(Proc_Channel)
        self.shortcut = Sequential()
        if DownSample == 1:
            if Proc_Channel != 64:
                in_channels = int(Proc_Channel/2)
                stride = 2
            self.shortcut = Sequential(Conv2d(in_channels = int(in_channels), 
                                   out_channels = int(Proc_Channel), 
                                   kernel_size = 3, 
                                   stride = int(stride), 
                                   padding = 1),
                               BatchNorm2d(Proc_Channel)) 
        self.ConvLayer1 = Sequential(Conv2d(in_channels, Proc_Channel, 3, stride, 1),
                            BatchNorm2d(Proc_Channel))
        self.ConvLayer2 = Sequential(Conv2d(Proc_Channel, Proc_Channel, 3, 1, 1),
                            BatchNorm2d(Proc_Channel)) 
    def forward(self, x):
        Residual = self.shortcut(x)
        x = self.ConvLayer1(x)
        x = F.relu(x)
        x = self.ConvLayer2(x)
        x = F.relu(x)
        x = x + Residual
        x = F.relu(x)
        return x
    
class Bottle_neck(nn.Module):
    def __init__(self, Proc_Channel, DownSample):
        super(Bottle_neck,self).__init__()
        Proc_Channel = int(Proc_Channel)
        stride = 1
        in_channels = int(Proc_Channel * 4)
        if DownSample == 1:
            if Proc_Channel == 64:
                in_channels = Proc_Channel
            else:
                stride = 2
                in_channels = int(Proc_Channel * 2)
        self.ConvLayer1 = Sequential(Conv2d(in_channels, Proc_Channel, 1, stride, 0),
                            BatchNorm2d(Proc_Channel), 
                            ReLU())
        self.ConvLayer2 = Sequential(Conv2d(Proc_Channel, Proc_Channel, 3, stride, 1),
                            BatchNorm2d(Proc_Channel), 
                            ReLU())
        self.ConvLayer3 = Sequential(Conv2d(Proc_Channel, Proc_Channel * 4, 1, stride, 0),
                            BatchNorm2d(Proc_Channel * 4), 
                            ReLU())
        self.shortcut = Sequential(Conv2d(in_channels, Proc_Channel * 4, 3, stride, 1),
                          BatchNorm2d(Proc_Channel * 4))
    def forward(self,x):
        Residual = self.shortcut(x)
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = x + Residual
        return x

class Resnet(Module):
    arch = {18 : [Basic_Block, [2, 2, 2, 2], 512], 
             34 : [Basic_Block, [3, 4, 6, 3], 512], 
            50 : [Bottle_neck, [3, 4, 6, 3], 2048], 
            101 : [Bottle_neck, [3, 4, 23, 3], 2048], 
            152 : [Bottle_neck, [3, 8, 36, 3], 2048]}
    def __init__(self,typ):
        super(Resnet,self).__init__()
        [block, layer_arch, final_channel] = Resnet.arch[typ]
        self.Resize = Resize((224,224), interpolation = InterpolationMode.BILINEAR)
        self.final_channel = final_channel
        self.stem = Sequential(Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3),
                        BatchNorm2d(64),
                        ReLU(),
                        MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        self.stage1 = self.make_layer(block, 64, layer_arch[0], stride = 1)
        self.stage2 = self.make_layer(block, 128, layer_arch[1], stride = 2)
        self.stage3 = self.make_layer(block, 256, layer_arch[2], stride = 2)
        self.stage4 = self.make_layer(block, 512, layer_arch[3], stride = 2)
        self.Aver_pool = AdaptiveAvgPool2d((1,1))
        self.fc = Linear(self.final_channel, 10)
        self.softmax = Softmax(dim = 1)
    def make_layer(self, block, Proc_Channel, layer_arch, stride):
        layer = []
        for i in range(layer_arch):
            if i ==0 :
                layer.append(block(Proc_Channel,True))
            else:
                layer.append(block(Proc_Channel,False))
        return Sequential(*layer)
    def forward(self,x):
        x = self.Resize(x)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.Aver_pool(x)
        x = x.view(-1, self.final_channel)
        x = self.fc(x)
        x = F.relu(x)
        x = self.softmax(x)
        return x
def Resnet18():
    return Resnet(Basic_Block, [2,2,2,2], 512)
def Resnet34():
    return Resnet(Basic_Block, [3,4,6,3], 512)
def Resnet50():
    return Resnet(Bottle_neck, [3,4,6,3] , 2048)
def Resnet101():
    return Resnet(Bottle_neck, [3,4,23,3] , 2048)
def Resnet152():
    return Resnet(Bottle_neck, [3,8,36,3] , 2048)