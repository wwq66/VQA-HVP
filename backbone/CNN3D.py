import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
     
       
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
       # print(out.shape)# [B,240,1,1]
        return self.sigmoid(out)


class CNN3D(nn.Module):
    def __init__(self, t_dim=80, img_x=64, img_y=64, drop_p=0.2, fc_hidden1=1024, fc_hidden2=128):

        super(CNN3D, self).__init__()
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        
        self.drop_p = drop_p
       # self.num_classes = num_classes
        self.ch1, self.ch2,self.ch3 = 32, 32,128
        
        self.k1, self.k2,self.k3 = (1, 1, 1), (3, 3, 3) ,(1, 1, 1) # 3d kernel size
        self.s1, self.s2= (1, 1, 1), (2, 2, 2)   # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding
        
        # compute conv1 & conv2 output shape
#         self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
#         self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
   
        def get_layer(in_size,out_size):
            layer = nn.Sequential(
                nn.Conv3d(in_channels=int(in_size), out_channels=out_size//4, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
                nn.BatchNorm3d(out_size//4),
                
                nn.Conv3d(in_channels=out_size//4, out_channels=out_size//4, kernel_size=self.k2, stride=self.s2,padding=self.pd2),
                nn.BatchNorm3d(out_size//4),
                
                nn.Conv3d(in_channels=out_size//4, out_channels=out_size, kernel_size=self.k3, stride=self.s1,padding=self.pd1),
                nn.BatchNorm3d(out_size),
                
            )
            return layer
            
        self.botten1 = get_layer(int(1),int(128))
        self.botten2 = get_layer(128,256)
        self.botten3 = get_layer(256,512)
        self.botten4 = get_layer(512,1024)
        self.botten5 = get_layer(1024,2048)
        self.pool3D = nn.AdaptiveMaxPool3d((1,1,1))
        
        self.fc1 = nn.Linear(2048, self.fc_hidden1) 
        self.fc2 = nn.Linear(self.fc_hidden1, 1)  
        
        
        
    def forward(self, input, input_length):
      
        
        input = input.reshape((-1,1,input.shape[1],64,64))#     [8,1,8000,64,64]
   #     print("input_1",input.shape,input_length.shape) #[8, 8000, 4096]
        input = self.botten1(input)
   #     print("input_2",input.shape) # [1, 128, 3999, 31, 31]
        
        input = self.botten2(input)
    #    print("input_3",input.shape)  #[[1, 256, 1999, 15, 15]]
        
        input = self.botten3(input)
   #     print("input_4",input.shape)  #[1, 512, 999, 7, 7]
        
        
        input = self.botten4(input)
    #    print("input_5",input.shape)  #[1, 1024, 499, 3, 3]
        
        input = self.botten5(input)
     #   print("input_6",input.shape)
#         x = input.view(input.size(0), -1)
#         print("input_7",x.shape)
#         x = torch.unsqueeze(x, 0)
        x = self.pool3D(input)
     #   print("input_7",x.shape)
        x = x.reshape((1,2048))
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.squeeze(0)
        #print("input_8",x.shape)
    
        return x 
    
    
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes, stride= 1) :
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes,kernel_size=1, stride=stride, bias=False)

  

class ANN(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)  #
        self.dropout = nn.Dropout(p=dropout_p)  #
        self.fc = nn.Linear(reduced_size, reduced_size)  #

    def forward(self, input):
        input = self.fc0(input)  # linear
        for i in range(self.n_ANNlayers-1):  # nonlinear
            input = self.fc(self.dropout(F.relu(input)))
        return input


def TP(q, tau=12, beta=0.5):
    """subjectively-inspired temporal pooling"""
    q = torch.unsqueeze(torch.t(q), 0)
    qm = -float('inf')*torch.ones((1, 1, tau-1)).to(q.device)
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
    l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l


def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape




class ResNet3D(nn.Module):

    def __init__(
        self,
        block,
        layers,
        num_classes = 1,
        zero_init_residual= False,
        groups = 1,
        width_per_group = 64,
        replace_stride_with_dilation = None,
        norm_layer = None
    ) :
        
        super(ResNet3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if replace_stride_with_dilation is None:

            replace_stride_with_dilation = [False, False, False]
        
      #  self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=7, stride=2, padding=3)
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)        
        
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,dilate=replace_stride_with_dilation[0])
        
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,dilate=replace_stride_with_dilation[2])
        
        self.ca1 = ChannelAttention(240)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks,stride=1, dilate = False) :
        
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
       
        x = x.reshape((-1,240,64,64))
    #    print(x.shape)
        x = self.ca1(x)*x
        
        x = x.reshape((-1,1,x.shape[1],64,64))#
    #    print("input_1",x.shape) #[1, 1, 240, 64, 64]
    #[1,8,30,24,24]
        
        x = self.conv1(x)
    #    print("input_2",x.shape) #[1, 64, 361, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
    #    print("input_3",x.shape) #[1, 64, 361, 32, 32]
        x = self.maxpool(x)
    #    print("input_4",x.shape) #[1, 64, 181, 16, 16]
        x = self.layer1(x)
                                     
    #    print("input_5",x.shape) # [1, 256, 60, 16, 16]
        x = self.layer2(x)
   #     print("input_6",x.shape)# [1, 512, 91, 8, 8]
        x = self.layer3(x)
   #     print("input_7",x.shape)#[1, 1024, 46, 4, 4]
        x = self.layer4(x)
   #     print("input_8",x.shape)# [1, 2048, 23, 2, 2]
        x = self.avgpool(x)
   #     print("input_9",x.shape)# [1, 2048, 1, 1, 1]
        x = torch.flatten(x, 1)
    #    print("input_9",x.shape)#[1, 2048]
        x = self.fc(x)
     #   print("input_10",x.shape)# [1, 1]
        x = x.squeeze(0)
        return x

    def forward(self, x) :
        return self._forward_impl(x)
    

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups= 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None
    ) :
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        

        out = self.conv1(x)
    
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
       
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
    
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)  
    

class FPN_Dliated_ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""
    def __init__(self):
        super(FPN_Dliated_ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=False).children()))
        for p in self.features.parameters():
            p.requires_grad = False
        self.all_features = {}
    def forward(self, x):
        # features@: 7->res5c
        
        for ii, model in enumerate(self.features):
            x = model(x)
            
            if ii == 4:
                f1 = x    # [length,256,w/4,h/4]
                f1 = nn.functional.adaptive_avg_pool2d(f1, (16,16))
                self.all_features["down_samplt_256"] = f1
              #  print("f1:",f1.shape)
            if ii == 5:
                f2 = x    # [length,_512,w/8,h/8]
                f2 = nn.functional.adaptive_avg_pool2d(f2, (8,8))
                self.all_features["down_samplt_512"] = f2
             #   print("f2:",f2.shape)
            if ii == 6:
                f3 = x    # [length,1024,w/16,h/4]
                f3 = nn.functional.adaptive_avg_pool2d(f3, (4,4))
                self.all_features["down_samplt_1024"] = f3
             #   print("f3:",f3.shape)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                features_std = global_std_pool2d(x)
                self.all_features["down_samplt_2048"] = [features_mean, features_std]
            #print([features_mean, features_std])
                return self.all_features


class FPN_ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""
    def __init__(self):
        super(FPN_ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=False).children()))
        for p in self.features.parameters():
            p.requires_grad = False
        self.all_features = {}
    def forward(self, x):
        # features@: 7->res5c
        
        for ii, model in enumerate(self.features):
            x = model(x)
            
            if ii == 4:
                f1 = x    # [length,256,w/4,h/4]
                f1 = nn.functional.adaptive_avg_pool2d(f1, (16,16))
                self.all_features["down_samplt_256"] = f1
              #  print("f1:",f1.shape)
            if ii == 5:
                f2 = x    # [length,_512,w/8,h/8]
                f2 = nn.functional.adaptive_avg_pool2d(f2, (8,8))
                self.all_features["down_samplt_512"] = f2
             #   print("f2:",f2.shape)
            if ii == 6:
                f3 = x    # [length,1024,w/16,h/4]
                f3 = nn.functional.adaptive_avg_pool2d(f3, (4,4))
                self.all_features["down_samplt_1024"] = f3
             #   print("f3:",f3.shape)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                features_std = global_std_pool2d(x)
                self.all_features["down_samplt_2048"] = [features_mean, features_std]
            #print([features_mean, features_std])
                return self.all_features


def add_FPN(x,x_input):
    out_channel = x.shape[2]
    down_1x1=conv1x1(x_input.shape[1],out_channel)
    x_input =x_input.float()
    
    x_add = down_1x1(x_input)
    x_add = x_add.permute(0,2,1,3,4).to("cuda")
    x = x+x_add
    #print("x_add2",x_add.shape,x.shape)
    return x

            
class FPN_ResNet3D(nn.Module):

    def __init__(
        self,
        block,
        layers,
        num_classes = 1,
        zero_init_residual= False,
        groups = 1,
        width_per_group = 64,
        replace_stride_with_dilation = None,
        norm_layer = None
    ) :
        
        super(FPN_ResNet3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if replace_stride_with_dilation is None:

            replace_stride_with_dilation = [False, False, False]
        
      #  self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=7, stride=2, padding=3)
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)        
        
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,dilate=replace_stride_with_dilation[0])
        
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks,stride=1, dilate = False) :
        
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x_all):
        # See note [TorchScript super()]
        #torch.Size([200, 4096])
                # torch.Size([200, 256, 16, 16])
                # torch.Size([200, 512, 8, 8])
                # torch.Size([200, 1024, 4, 4])
        x = x_all[0]
#         print(x.shape)
#         print(x_all[1][0].shape)

        

        x = x.reshape((-1,1,x.shape[1],64,64))#
    #    print("input_1",x.shape) #[1, 1, 722, 64, 64]
        x = self.conv1(x)
    #    print("input_2",x.shape) #[1, 64, 361, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
    #    print("input_3",x.shape) #[1, 64, 361, 32, 32]
        x = self.maxpool(x)
    #    print("input_4",x.shape) #[1, 64, 181, 16, 16]
        x = self.layer1(x)

        x = add_FPN(x,x_all[1][0])

        x = self.layer2(x)
        
        x = add_FPN(x,x_all[1][1])
     #   print("x_add2",x.shape)
        
        
        
   #     print("input_6",x.shape)# [1, 512, 91, 8, 8]
        x = self.layer3(x)
        x = add_FPN(x,x_all[1][2])
      #  print("x_add2",x.shape)
        
   #     print("input_7",x.shape)#[1, 1024, 46, 4, 4]
        x = self.layer4(x)
   #     print("input_8",x.shape)# [1, 2048, 23, 2, 2]
        x = self.avgpool(x)
   #     print("input_9",x.shape)# [1, 2048, 1, 1, 1]
        x = torch.flatten(x, 1)
    #    print("input_9",x.shape)#[1, 2048]
        x = self.fc(x)
     #   print("input_10",x.shape)# [1, 1]
        x = x.squeeze(0)
        return x

    def forward(self, x) :
        return self._forward_impl(x)


class Dliated_ResNet3D(nn.Module):

    def __init__(
        self,
        block,
        layers,
        num_classes = 1,
        zero_init_residual= False,
        groups = 1,
        width_per_group = 64,
        replace_stride_with_dilation = None,
        norm_layer = None
    ) :
        
        super(Dliated_ResNet3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if replace_stride_with_dilation is None:

            replace_stride_with_dilation = [False, False, False]
        
      #  self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.conv1 = nn.Conv3d(2, self.inplanes, kernel_size=7, stride=2, padding=3)
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)        
        
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,dilate=replace_stride_with_dilation[0])
        
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks,stride=1, dilate = False) :
        
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x,d_x):
        # See note [TorchScript super()]
        #torch.Size([200, 4096])
                # torch.Size([200, 256, 16, 16])
                # torch.Size([200, 512, 8, 8])
                # torch.Size([200, 1024, 4, 4])

#         print(x.shape)
#         print(x_all[1][0].shape)

        

        x = x.reshape((-1,1,x.shape[1],64,64))#
        d_x = d_x.reshape((-1,1,d_x.shape[1],64,64))
        x = torch.cat((x,d_x),1)
     #   print(x.shape)
        
        
    #    print("input_1",x.shape) #[1, 1, 722, 64, 64]
        x = self.conv1(x)
    #    print("input_2",x.shape) #[1, 64, 361, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
    #    print("input_3",x.shape) #[1, 64, 361, 32, 32]
        x = self.maxpool(x)
    #    print("input_4",x.shape) #[1, 64, 181, 16, 16]
        x = self.layer1(x)

        x = self.layer2(x)

     #   print("x_add2",x.shape)
          
   #     print("input_6",x.shape)# [1, 512, 91, 8, 8]
        x = self.layer3(x)
  
      #  print("x_add2",x.shape)
        
   #     print("input_7",x.shape)#[1, 1024, 46, 4, 4]
        x = self.layer4(x)
   #     print("input_8",x.shape)# [1, 2048, 23, 2, 2]
        x = self.avgpool(x)
   #     print("input_9",x.shape)# [1, 2048, 1, 1, 1]
        x = torch.flatten(x, 1)
    #    print("input_9",x.shape)#[1, 2048]
        x = self.fc(x)
     #   print("input_10",x.shape)# [1, 1]
        x = x.squeeze(0)
        return x

    def forward(self, x,d_x) :
        return self._forward_impl(x,d_x)

    
    
    

def make_down(sam_ratio):
    x = list()
    for j in range(sam_ratio):
        a = 2 ** (j + 1)
        p = 0
        y = list()
        for i in range(0,240, a):
            p += 1
            y.append(i)
        x.append(y)


  
    frames = x
    return frames
    
    
class FPN_Dliated_ResNet3D(nn.Module):

    def __init__(
        self,
        block,
        layers,
        num_classes = 1,
        zero_init_residual= False,
        groups = 1,
        width_per_group = 64,
        replace_stride_with_dilation = None,
        norm_layer = None
    ) :
        
        super(FPN_Dliated_ResNet3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.sam_ratio = 3
        
        self.groups = groups
        self.base_width = width_per_group
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if replace_stride_with_dilation is None:

            replace_stride_with_dilation = [False, False, False]
        
      #  self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.conv1 = nn.Conv3d(2, self.inplanes, kernel_size=7, stride=2, padding=3)
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)        
        
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        
        self.ca1 = ChannelAttention(240)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,dilate=replace_stride_with_dilation[0])
        
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        
        
        self.frames = make_down(self.sam_ratio)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks,stride=1, dilate = False) :
        
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x,d_x):
        # See note [TorchScript super()]
        #torch.Size([200, 4096])
                # torch.Size([200, 256, 16, 16])
                # torch.Size([200, 512, 8, 8])
                # torch.Size([200, 1024, 4, 4])

#         print(x.shape)
#         print(x_all[1][0].shape)

        x = x.reshape((-1,240,64,64))
    #    print(x.shape)
        x = self.ca1(x)*x

        x = x.reshape((-1,1,x.shape[1],64,64))#
        
        
        d_x = d_x.reshape((-1,240,64,64))
    #    print(x.shape)
        d_x = self.ca1(d_x)*d_x
        
        
        d_x = d_x.reshape((-1,1,d_x.shape[1],64,64))
        x = torch.cat((x,d_x),1)
#         print(x.shape)
       
       
        out_down_1 = x[:,:, self.frames[0],:, :]
        out_down_2 = x[:,:, self.frames[1],:, :]
        out_down_3 = x[:,:, self.frames[2],:, :]
        print("out_down_1",out_down_1.shape,out_down_2.shape,out_down_3.shape)
       
        
    #    print("input_1",x.shape) #[1, 1, 722, 64, 64]
        x = self.conv1(x)
#         print("input_2",x.shape) #[1, 64, 361, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
    #    print("input_3",x.shape) #[1, 64, 361, 32, 32]
        x = self.maxpool(x)
    #    print("input_4",x.shape) #[1, 64, 181, 16, 16]
        x = self.layer1(x)

        x = self.layer2(x)

     #   print("x_add2",x.shape)
          
   #     print("input_6",x.shape)# [1, 512, 91, 8, 8]
        x = self.layer3(x)
  
      #  print("x_add2",x.shape)
     #   sys.exit()
   #     print("input_7",x.shape)#[1, 1024, 46, 4, 4]
        x = self.layer4(x)
   #     print("input_8",x.shape)# [1, 2048, 23, 2, 2]
        x = self.avgpool(x)
   #     print("input_9",x.shape)# [1, 2048, 1, 1, 1]
        x = torch.flatten(x, 1)
    #    print("input_9",x.shape)#[1, 2048]
        x = self.fc(x)
     #   print("input_10",x.shape)# [1, 1]
        x = x.squeeze(0)
        return x

    def forward(self, x,d_x) :
        return self._forward_impl(x,d_x)

    
    
class FPN_Dliated_LOSS_ResNet3D(nn.Module):

    def __init__(
        self,
        block,
        layers,
        num_classes = 1,
        zero_init_residual= False,
        groups = 1,
        width_per_group = 64,
        replace_stride_with_dilation = None,
        norm_layer = None
    ) :
        
        super(FPN_Dliated_LOSS_ResNet3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.sam_ratio = 3
        
        self.groups = groups
        self.base_width = width_per_group
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if replace_stride_with_dilation is None:

            replace_stride_with_dilation = [False, False, False]
        
      #  self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.conv1 = nn.Conv3d(2, self.inplanes, kernel_size=7, stride=2, padding=3)
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)        
        
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        
        self.ca1 = ChannelAttention(240)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,dilate=replace_stride_with_dilation[0])
        
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(1024, num_classes)
        
        self.frames = make_down(self.sam_ratio)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks,stride=1, dilate = False) :
        
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def end_layer(self,x):
        x = self.conv1(x)
#         print("input_2",x.shape) #[1, 64, 120, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
    #    print("input_3",x.shape) #[1, 64, 120, 32, 32]
        x = self.maxpool(x)
    #    print("input_4",x.shape) #[1, 64, 60, 16, 16]
        x = self.layer1(x)
        
       # print("input_5",x.shape) # [1, 256, 60, 16, 16]
        x = self.layer2(x)

     #   print("x_add2",x.shape)
          
   #     print("input_6",x.shape)# [1, 512, 91, 8, 8]
        x = self.layer3(x)
  
      #  print("x_add2",x.shape)
     #   sys.exit()
   #     print("input_7",x.shape)#[1, 1024, 46, 4, 4]
        x = self.layer4(x)
   #     print("input_8",x.shape)# [1, 2048, 23, 2, 2]
        x = self.avgpool(x)
   #     print("input_9",x.shape)# [1, 2048, 1, 1, 1]
        x = torch.flatten(x, 1)
    #    print("input_9",x.shape)#[1, 2048]
        x = self.fc(x)
     #   print("input_10",x.shape)# [1, 1]
        x = x.squeeze(0)
        
        return x

    def _forward_impl(self, x,d_x):
        
        x = x.reshape((-1,240,64,64))
    #    print(x.shape)
        x = self.ca1(x)*x

        x = x.reshape((-1,1,x.shape[1],64,64))#
        
        
        d_x = d_x.reshape((-1,240,64,64))
    #    print(x.shape)
        d_x = self.ca1(d_x)*d_x
        
        
        d_x = d_x.reshape((-1,1,d_x.shape[1],64,64))
        x = torch.cat((x,d_x),1)
#         print(x.shape)
       
       
        out_down_1 = x[:,:, self.frames[0],:, :]
        out_down_2 = x[:,:, self.frames[1],:, :]
        out_down_3 = x[:,:, self.frames[2],:, :]
       # print("out_stage_0",out_down_1.shape,out_down_2.shape,out_down_3.shape)
        
        
        out_down_1 = self.conv1(out_down_1)
        out_down_1 = self.bn1(out_down_1)
        out_down_1 = self.relu(out_down_1)
        out_down_1 = self.maxpool(out_down_1)
        
        out_down_2 = self.conv1(out_down_2)
        out_down_2 = self.bn1(out_down_2)
        out_down_2 = self.relu(out_down_2)     
        out_down_2 = self.maxpool(out_down_2)
        
    
        out_down_3 = self.conv1(out_down_3)
        out_down_3 = self.bn1(out_down_3)
        out_down_3 = self.relu(out_down_3)
        out_down_3 = self.maxpool(out_down_3)
   #     print("out_stage_1",out_down_1.shape,out_down_2.shape,out_down_3.shape)torch.Size([8, 64, 30, 16, 16]) torch.Size([8, 64, 15, 16, 16]) torch.Size([8, 64, 8, 16, 16])

        out_down_1 = self.layer1(out_down_1)
        out_down_2 = self.layer1(out_down_2)
        out_down_3 = self.layer1(out_down_3)
  #      print("out_stage_2",out_down_1.shape,out_down_2.shape,out_down_3.shape) torch.Size([8, 256, 30, 16, 16]) torch.Size([8, 256, 15, 16, 16]) torch.Size([8, 256, 8, 16, 16])

        out_down_1 = self.layer2(out_down_1)
        out_down_2 = self.layer2(out_down_2)
        out_down_3 = self.layer2(out_down_3)
    #    print("out_stage_3",out_down_1.shape,out_down_2.shape,out_down_3.shape)torch.Size([8, 512, 15, 8, 8]) torch.Size([8, 512, 8, 8, 8]) torch.Size([8, 512, 4, 8, 8])
    
        out_down_2 = self.layer3(out_down_2)
        out_down_3 = self.layer3(out_down_3)
    #    print("out_stage_4",out_down_1.shape,out_down_2.shape,out_down_3.shape)torch.Size([8, 512, 15, 8, 8]) torch.Size([8, 1024, 4, 4, 4]) torch.Size([8, 1024, 2, 4, 4])
    
        out_down_3 = self.layer4(out_down_3)
     #   print("out_stage_5",out_down_1.shape,out_down_2.shape,out_down_3.shape)torch.Size([8, 512, 15, 8, 8]) torch.Size([8, 1024, 4, 4, 4]) torch.Size([8, 2048, 1, 2, 2])
        
        
        
        up_1 = torch.nn.functional.interpolate(out_down_1, (30,8,8), mode='nearest')
        up_2 = torch.nn.functional.interpolate(out_down_2, (15,4,4), mode='nearest')
        up_3 = torch.nn.functional.interpolate(out_down_3, (8,2,2), mode='nearest')
        
            
  #      print("up_3",up_3.shape)# ([8, 256, 30, 32, 32])
        
        
        x = self.conv1(x)
       # print("input_2",x.shape) #[1, 64, 120, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
    #    print("input_3",x.shape) #[1, 64, 120, 32, 32]
        x = self.maxpool(x)
    #    print("input_4",x.shape) #[1, 64, 60, 16, 16]
        x = self.layer1(x)

        x = self.layer2(x)
        
     #   print("x_add2",x.shape)
        out_1 = x+up_1
    #    x = out_1
        out_1 = self.avgpool(out_1)
        out_1 = torch.flatten(out_1, 1)
        out_1 = self.fc2(out_1)
        socre1 = out_1.squeeze(0)
 #       print("out_1",out_1.shape)# [8,1]
        
        
        x = self.layer3(x)
  
      #  print("x_add2",x.shape)
        
      #  print("input_7",x.shape)#[1, 1024, 46, 4, 4]
        out_2 = x+up_2
     #   x = out_2
        out_2 = self.avgpool(out_2)
        out_2 = torch.flatten(out_2, 1)
        out_2 = self.fc3(out_2)
        socre2 = out_2.squeeze(0)
  #      print("out_2",socre2.shape)# [8,1]
        
        
        
        x = self.layer4(x)
   #     print("input_8",x.shape)# [1, 2048, 23, 2, 2]
        out_3 = x+up_3
   #     x = out_3
        out_3 = self.avgpool(out_3)
        out_3 = torch.flatten(out_3, 1)
        out_3 = self.fc(out_3)
        socre3 = out_3.squeeze(0)
    #    print("out_3",socre3.shape)# [8,1]
        
        
        x = self.avgpool(x)
   #     print("input_9",x.shape)# [1, 2048, 1, 1, 1]
       
    
    
        x = torch.flatten(x, 1)
       # print("input_9",x.shape)#[1, 2048]
        
        
        
        x = self.fc(x)
     #   print("input_10",x.shape)# [1, 1]
        x = x.squeeze(0)
        
     
     #   print("input_1",x.shape) #[1, 1, 240, 64, 64]
        score = x
    #    print(self.end_layer(x),self.end_layer(out_down_1))+0.5*socre1+0.3*socre2+0.2*,socre3
    
        return score,socre1,socre2,socre3

    def forward(self, x,d_x) :
        return self._forward_impl(x,d_x)
