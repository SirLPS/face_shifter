# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:17:02 2020

@author: lenovo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.1))
    def forward(self, x):
        return self.conv(x)
    
    
class ConvTrans(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_trans = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.1))
    def forward(self, x):
        return self.conv_trans(x)
    
    
class BilinearUp(nn.Module):
    """
    nn.Upsampling is deprecated. Use nn.functional.interpolate instead
    """
    def __init__(self, scale_factor=2):
        super().__init__()
        self.mode = 'bilinear'
        self.scale_factor = scale_factor 
    def forward(self, x):
        return nn.functional.interpolate(input=x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
    
    
class AAD(nn.Module):
    def __init__(self, in_channels, att_channels):
        super(AAD,self).__init__() 
        self.conv1 = nn.Conv2d(att_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(att_channels, in_channels, 3, 1, 1)
        self.linear1 = nn.Linear(512, in_channels, bias=True)    # should be 256 as orignal
        self.linear2 = nn.Linear(512, in_channels, bias=True)
        self.conv_sigmoid = nn.Sequential(nn.Conv2d(in_channels, 1,3,1,1),
                                          nn.Sigmoid())
    
    def forward(self, H_in, Z_aat, Z_id):
        N, C, H, W = H_in.shape
        N_a, C_a, H_a, W_a = Z_aat.shape
        N_id, C_id, H_id, W_id  = Z_id.shape
#        print(H_in.shape, Z_aat.shape, Z_id.shape)
        H_k = nn.InstanceNorm2d(num_features=C)(H_in)
        Gama_att = self.conv1(Z_aat)
        Beta_att = self.conv2(Z_aat)
        A = H_k*Gama_att + Beta_att
        M = self.conv_sigmoid(H_k)
        Gama_id = self.linear1(Z_id.squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
        Beta_id = self.linear2(Z_id.squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
        I = H_k*Gama_id + Beta_id
        H_out = (1-M)*A + M*I
        return H_out


class AAD_ResBlk(nn.Module):
    def __init__(self, in_channels, out_channels, att_channels):   
        super(AAD_ResBlk, self).__init__() 
        self.flag = True if in_channels==out_channels else False
        self.AAD_1 = AAD(in_channels, att_channels)
        self.Relu_conv1 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1))

        self.AAD_2 = AAD(out_channels, att_channels)
        self.Relu_conv2 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        
        self.AAD_3 = AAD(in_channels, att_channels)
        self.Relu_conv3 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        
    def forward(self, H_in, Z_aat, Z_id):
        x1 = self.Relu_conv1(self.AAD_1(H_in, Z_aat, Z_id))
        x2 = self.Relu_conv2(self.AAD_2(x1, Z_aat, Z_id))
        if not self.flag:
            x3 = self.Relu_conv3(self.AAD_3(H_in, Z_aat, Z_id))
#            print(x2.shape, x3.shape)
            return x3 + x2
        return H_in + x2 
    
    
# Define the PatchGAN discriminator 
class NLayerDiscriminator(nn.Module):
    """  pix2pixHD
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        self.n_layers = n_layers
        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
       
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf*2, 512)
            sequence += [[
                    nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),  
                    norm_layer(nf),
                    nn.LeakyReLU(0.2, True)
                    ]]

        nf_prev = nf
        nf = min(nf*2, 512)
        sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
                ]]
           
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]
            
        sequence_stream = []
        for n in range(len(sequence)):
            sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)
            
    def forward(self, input):
        return self.model(input)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False, num_D=3):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            setattr(self, 'layer'+str(i), netD.model)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1,1], count_include_pad=False)

    def singleD_forward(self, model, input):
        return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        
        for i in range(num_D):
            model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i!=(num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    MSGGan = False
    if  MSGGan:
        alpha = torch.rand(1, 1)
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = [alpha * rd + ((1 - alpha) * fd) for rd, fd in zip(real_data, fake_data)]
        interpolates = [i.to(device) for i in interpolates]
        interpolates = [torch.autograd.Variable(i, requires_grad=True) for i in interpolates]

        disc_interpolates = netD(interpolates)
    else:
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)#.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)[0][0]

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
