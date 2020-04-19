# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 17:14:50 2020

@author: lenovo
"""

import torch.nn.functional as F
from utils import *
import torch 
from network import HEARNet, Att_Encoder, AAD_Gen
from metric import loss_hinge_dis, loss_hinge_gen, IdLoss, ChgLoss, RecLoss
import argparse
import os
from get_feats import load_model, ArcFace_Net
from dataloader import aug_data_loader


def train(args):
    # gpu init
    multi_gpu = False
    if len(args.gpus.split(',')) > 1:
        multi_gpu = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    G = AAD_Gen()
    F = ArcFace_Net(args.backbone, args.test_model_path)  # no need to train
    E = Att_Encoder()
    H = HEARNet()
    
    G = load_model(G, 'path_to_G_model')
    E = load_model(E, 'path_to_E_model')
    
    G.eval()
    E.eval()
    
    optimizer = torch.optim.Adam({'params': H.parameters()},
                                   lr=0.0004, betas=(0.0, 0.999))
    
    if multi_gpu:
        H = DataParallel(D).to(device)
    else:
        H = D.to(device)
        

    for epoch in range(1, args.total_epoch+1):    
        H.train()
#        F.test()      Only extract features!  # input dim=3,256,256   out dim=256 ! 
        for step, data in enumerate(aug_data_loader):
            try:
                img, label = data
            except Exception as e:
                continue
            source = img[:4,:,:,:].to(device)
            target = img[[0,1,2,4],:,:,:].to(device)
    
    
            Y_tt = G(F(target), E(target))
            error = target - Y_tt
            Yst0 = G(F(source), E(target))
            Yst = H(torch.cat((Yst0, error), dim=1))
        
            optimizer.zero_grad() 

            L_id = IdLoss()(F(Yst), F(source))
            L_chg = ChgLoss()(Yst0, Yst)
            L_rec = RecLoss()(Yst0[:-1,:,:,:], target[:-1,:,:,:], label)
            
            Loss = (L_id + L_chg + L_rec).to(device)
            Loss.backward()
            optimizer.step()
            
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch for FaceShifter')
    parser.add_argument('--backbone', type=str, default='resnet50', help='resnet18, resnet50, resnet101, resnet152')
    parser.add_argument('--test_model_path', type=str, default='', help='path to arcface pretrained model')

    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension, 256 or 512. original is 256 !!!')
    parser.add_argument('--scale_size', type=float, default=32.0, help='scale size')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--total_epoch', type=int, default=18, help='total epochs')

    parser.add_argument('--save_freq', type=int, default=3000, help='save frequency')
    parser.add_argument('--test_freq', type=int, default=3000, help='test frequency')
    parser.add_argument('--resume', type=int, default=False, help='resume model')
    parser.add_argument('--net_path', type=str, default='', help='resume model')
    parser.add_argument('--margin_path', type=str, default='', help='resume model')
    parser.add_argument('--save_dir', type=str, default='./model', help='model save dir')
    parser.add_argument('--model_pre', type=str, default='SERES100_', help='model prefix')
    
    parser.add_argument('--gpus', type=str, default='0', help='model prefix')
    
    args = parser.parse_args()
    train(args)
    