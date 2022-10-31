import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from torchvision import transforms
from torch import tensor


class BB_model(nn.Module):
    def __init__(self):
        super(BB_model, self).__init__()
        resnet = models.resnet34(pretrained=True)
        #resnet = models.resnet34(weights='ResNet34_Weights.DEFAULT')
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x), self.bb(x)

class sigmoid_bb(nn.Module):
    def __init__(self,pretrained_model=models.resnet18(), n_features=512, im_width = 256, im_height =256):
        """takes pretrained model, outputs either classes or feature map
        ----Params----
        pretrained_model: pretrained_network, default models.resnet18(weights='IMAGENET1K_V1')
        fixed_backbone: whether to propagate gradient through backbone or not"""
        super(sigmoid_bb, self).__init__()
        self.pretrained_model = pretrained_model
        n_layers = len(list(pretrained_model.children()))
        layers = list(pretrained_model.children())[:n_layers-2]
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.BatchNorm1d(n_features), nn.Linear(n_features, 4))
        self.sigmoid = nn.Sigmoid()
        self.im_height = im_height
        self.im_width = im_width
        
    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1)
        bb =self.sigmoid(self.classifier(x))

        x_centre = bb[:,0]*self.im_width
        y_centre = bb[:,1]*self.im_height
        width = bb[:,2]*self.im_width
        height = bb[:,3]*self.im_height
        #mew format: [ymin,xmin,ymax,xmax]
        bb_pred = torch.vstack((y_centre-height/2,x_centre-width/2,y_centre+height/2,x_centre+width/2)).T

        return bb_pred
    
    def set_use_fc(self,use_fc):
        self.use_fc_layer = use_fc

class backbone_network(nn.Module):
    def __init__(self,n_classes,pretrained_model=models.resnet18(),fixed_backbone=False,use_fc_layer=True,output_features =False, n_features=512):
        """takes pretrained model, outputs either classes or feature map
        ----Params----
        pretrained_model: pretrained_network, default models.resnet18(weights='IMAGENET1K_V1')
        fixed_backbone: whether to propagate gradient through backbone or not"""
        super(backbone_network, self).__init__()
        self.pretrained_model = pretrained_model
        self.fixed_backbone = fixed_backbone
        self.output_features = output_features
        self.use_fc_layer = use_fc_layer
        n_layers = len(list(pretrained_model.children()))
        layers = list(pretrained_model.children())[:n_layers-2]
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.BatchNorm1d(n_features), nn.Linear(n_features, n_classes))
        
    def forward(self, x):
        if self.fixed_backbone:
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1)

        if self.use_fc_layer:
            y = self.classifier(x)
            if self.output_features:
                return y, x
            else:
                return y
        else:
            return x
    
    def set_use_fc(self,use_fc):
        self.use_fc_layer = use_fc

class CNN(nn.Module):
    def __init__(self,n_classes,input_channels=3):
        """ 
        n_classes: number of outputs from final fully connected layer
        input image size must be 28x28"""
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=input_channels,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output n classes
        self.out = nn.Linear(32 * 7 * 7, n_classes)
    def forward(self, x):
        x = transforms.Resize(size=(28,28))(x)
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output  

        