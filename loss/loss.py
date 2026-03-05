import os

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def Loss_intensity(vis, ir, image_fused, model='sep', vis_w=1, ir_w=1):
    assert (vis.size() == ir.size() == image_fused.size())
    if model == 'sep':
        ir_li = F.l1_loss(image_fused, ir)
        vis_li = F.l1_loss(image_fused, vis)
        li = ir_w * ir_li + vis_w * vis_li
        return li
    elif model == 'max':
        joint = torch.max(ir, vis)
        li = F.l1_loss(image_fused, joint)
        return li


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, img1, img2, image_fused=None):
        if image_fused == None:
            image_1_Y = img1[:, :1, :, :]
            image_2_Y = img2[:, :1, :, :]
            gradient_1 = self.sobelconv(image_1_Y)
            gradient_2 = self.sobelconv(image_2_Y)
            Loss_gradient = F.l1_loss(gradient_1, gradient_2)
            return Loss_gradient
        else:
            image_1_Y = img1[:, :1, :, :]
            image_2_Y = img2[:, :1, :, :]
            image_fused_Y = image_fused[:, :1, :, :]
            gradient_1 = self.sobelconv(image_1_Y)
            gradient_2 = self.sobelconv(image_2_Y)
            gradient_fused = self.sobelconv(image_fused_Y)
            gradient_joint = torch.max(gradient_1, gradient_2)
            Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
            return Loss_gradient


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self, x):
        weightx = self.weightx.to(x.dtype)
        weighty = self.weighty.to(x.dtype)
        sobelx = F.conv2d(x, weightx, padding=1)
        sobely = F.conv2d(x, weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)