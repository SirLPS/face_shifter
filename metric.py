# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:35:14 2020

@author: lenovo
"""
import torch.nn.functional as F
import torch.nn as nn
import torch

# Hinge Loss (BigGAN) 
class loss_hinge_dis(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,dis_fake, dis_real):
        loss_real, loss_fake = 0, 0
        if isinstance(dis_fake, list):
            for i in range(len(dis_fake)):
                loss_real += torch.mean(F.relu(1. - dis_real[i][0]))
                loss_fake += torch.mean(F.relu(1. + dis_fake[i][0]))
        else:
             loss_real = torch.mean(F.relu(1. - dis_real))
             loss_fake = torch.mean(F.relu(1. + dis_fake))
        loss_real = loss_real/len(dis_fake)
        loss_fake = loss_fake/len(dis_fake)
        return loss_real, loss_fake
    # def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
      # loss = torch.mean(F.relu(1. - dis_real))
      # loss += torch.mean(F.relu(1. + dis_fake))
      # return loss

class loss_hinge_gen(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,dis_fake):
        loss = 0
        if isinstance(dis_fake, list):
            for i in range(len(dis_fake)):
                loss += -torch.mean(dis_fake[i][0])
        else:
            loss = -torch.mean(dis_fake)
        loss = loss/len(dis_fake)
        return loss    



def loss_hinge_dis_mse(dis_fake, dis_real):
    loss_real, loss_fake = 0, 0
    target_real_label, target_fake_label=torch.Tensor([1.0]),torch.Tensor([0.0])

    for i in range(len(dis_real)):
        loss_real += nn.MSELoss()(dis_real[i][0], target_real_label.expand_as(dis_real[i][0]).type_as(dis_real[i][0]))
    for i in range(len(dis_fake)):
        loss_fake += nn.MSELoss()(dis_fake[i][0], target_fake_label.expand_as(dis_fake[i][0]).type_as(dis_fake[i][0]))

    loss_real = loss_real/len(dis_real)
    loss_fake = loss_fake/len(dis_fake)
    return loss_real, loss_fake


def loss_hinge_gen_mse(dis_fake):
    loss = 0
    target_real_label = torch.Tensor([1.0])
    if isinstance(dis_fake, list):
        for i in range(len(dis_fake)):
            loss += nn.MSELoss()(dis_fake[i][0], target_real_label.expand_as(dis_fake[i][0]).type_as(dis_fake[i][0]))
    else:
        loss = -torch.mean(dis_fake)
    loss = loss/len(dis_fake)
    return loss    



def loss_bce(input,label):
    loss  = 0
    b_size = input[0][0].shape[0]
    for i in range(len(input)):
        label_ = torch.full((b_size,1,input[i][0].shape[2], input[i][0].shape[3]),label, device='cuda')
        loss += nn.BCELoss()(input[i][0], label_)
    loss = loss/len(input)
    return loss
    


class IdLoss(nn.Module):
    """
    torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, warped_source, source):
        target = torch.ones(source.shape[0])
        target = target.type_as(source)
        loss = nn.CosineEmbeddingLoss()(warped_source, source,target)
        return loss
    
    
class AttrLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, attr_warped_target, attr_target):
        loss = 0
        if isinstance(attr_target, list):
            for i in range(len(attr_target)):
                loss += nn.MSELoss()(attr_warped_target[i], attr_target[i])
#                print(loss)
        else:
            loss = nn.MSELoss()(attr_warped_target, attr_target)
        loss = loss /2
        return loss
    
    
class RecLoss(nn.Module):
    """
    target==source
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, warped_source, target, label):
        loss = 0
        for i in range(len(label)):
            if label[i]==1:
                loss += nn.MSELoss()(warped_source[i], target[i])
        loss = loss / 2
        return loss


class ChgLoss(nn.Module):
    """
    target==source
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, yst0, yst):
        loss = 0
        loss = nn.L1Loss()(yst0, yst)
        return loss     
        
    
