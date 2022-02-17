import torch
from torch import nn
import torch.nn.functional as F
from unet_model import UNet


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def forward(self, coeff, full_res_input):

        '''
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''

        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)




class Zero_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = UNet(n_channels=3)
        self.app = ApplyCoeffs()
        
        
    def forward(self, x):
        inputs = F.interpolate(x, size=[128, 128],   mode='bilinear', align_corners=True)

        coeff = self.net(inputs)
        
        
        coeff = F.interpolate(coeff, [x.shape[2], x.shape[3]],   mode='bilinear', align_corners=True)
        
        # 1. Tucker  http://tensorly.org/stable/index.html
        # 2. PooL    https://pytorch.org/docs/master/generated/torch.nn.AdaptiveMaxPool2d.html?highlight=pool#torch.nn.AdaptiveMaxPool2d
        
        
        out = self.app(coeff, x)

        return  out     



net = Zero_Net()

data = torch.zeros(1, 3, 3840, 2160)

print(net(data).shape)