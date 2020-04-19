# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 09:51:27 2020

@author: lenovo
"""


import torch
from torch.nn import DataParallel
from network import Att_Encoder, AAD_Gen
from get_feats import ArcFace_Net
from dataloader import SupplyCollate
from metric import loss_hinge_dis, loss_hinge_gen, IdLoss, AttrLoss, RecLoss
from utils import MultiscaleDiscriminator
import torchvision.transforms as trans
import argparse
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import time
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import interpolate as downsample



def test(args):
    # gpu init
    multi_gpu = False
    if len(args.gpus.split(',')) > 1:
        multi_gpu = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#    D = MultiscaleDiscriminator(input_nc=3, ndf=64, n_layers=3, use_sigmoid=True)    # pix2pix use MSEloss
    G = AAD_Gen()
    F = ArcFace_Net(args.backbone, args.arc_model_path)  # no need to train
    E = Att_Encoder()
    
    
    if multi_gpu:
#        D = DataParallel(D).to(device)
        G = DataParallel(G).to(device)
        F = DataParallel(F).to(device)
        E = DataParallel(E).to(device)
    else:
#        D = D.to(device)
        G = G.to(device)
        F = F.to(device)
        E = E.to(device)
    
    if args.resume:
        if os.path.isfile(args.resume_model_path):
            print("Loading checkpoint {} from". format(args.resume_model_path))
            checkpoint = torch.load(args.resume_model_path)
#            args.start_epoch = checkpoint["epoch"]
#            D.load_state_dict(checkpoint["state_dict_D"])
            G.load_state_dict(checkpoint["state_dict_G"])
            E.load_state_dict(checkpoint["state_dict_E"])
#            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
#            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
#            optimizer_GE.load_state_dict(checkpoint['optimizer_GE'])
            print('Loaded sucessfully......')
        else:
            print('Cannot found checkpoint {}'.format(args.resume_model_path))
    else:
        args.start_epoch = 1
        
    def trans_batch(batch):
        batch = batch[:,:,50:-10, 30:-30]   # fit faces better  # bs,3,196,196
#        print(batch.shape)    
        t =  trans.Compose([trans.ToPILImage(), trans.Resize((112,112)), trans.ToTensor()])   
        bs = batch.shape[0]
        res = torch.ones(bs,3,112,112).type_as(batch)
        for i in range(bs):
            res[i] = t(batch[i].cpu())
        return res
      
    def print_with_time(string):
        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + string)
        
        
    def showimg(Yst0):
#        bs = Yst0.shape[0]
        t = trans.Compose([trans.ToPILImage()])
        imgs = np.array(t(Yst0.cpu()))
        plt.imshow(imgs[0])      
        plt.show()
        

    def l2_norm(input,axis=1):
        norm = torch.norm(input,2,axis,True)
        output = torch.div(input, norm)
        return output
    
    dataset = ImageFolder(args.data_path)  
# transform later!!!   transform=trans.Compose([trans.ToTensor])
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, sampler=None, collate_fn=SupplyCollate(dataset))

#    D.eval()
    G.eval()
    E.eval()
    F.eval()


    while True:
        try:
            data = iter(data_loader).next()
            break
        except Exception as StopIteration:
            break
        except  Exception as e:
            continue
        
                   
    img, label = data
    
    source = img[[0]].to(device)
    target = img[[1]].to(device)

    #Zid = F(trans_batch(source))  # bs, 512
    Zid = F(downsample(source[:,:,50:-10, 30:-30], size=(112,112))) #[:,::2]

    
    Zatt = E(target)     # list:8  each：bs,,,
    Yst0 = G(Zid, Zatt)  # bs,3,256,256
    
    F = trans.Compose([trans.ToPILImage()])
    Yst0 = F(Yst0.cpu()[0])
    source = F(source.cpu()[0])
    target = F(target.cpu()[0])

    return source, target, Yst0

            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch for FaceShifter')
    parser.add_argument('--backbone', type=str, default='resnet50', help='resnet18, resnet50, resnet101, resnet152')
    parser.add_argument('--arc_model_path', type=str, default='/media/a/HDD/lyfeng/Face_Proj/FaceShifter/src/model_ir_se50.pth', help='path to arcface pretrained model')

    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension, 256 or 512. original is 256 !!!')
    parser.add_argument('--data_path', type=str, default = '/media/a/HDD/lyfeng/Face_Proj/vgg_face_dataset/images')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--start_epoch', type=int, default=1, help= 'the start of epoch')
    parser.add_argument('--total_epoch', type=int, default=50, help='total epochs')
    parser.add_argument('--dis_times', type=int, default=5, help='how often update discriminators ')

    parser.add_argument('--log_interval', type=int, default=500, help='how many batches to wait before logging training status')
    parser.add_argument('--lr_step', type=int, default=10, help= 'lr decay step')
    parser.add_argument('--resume', type=int, default=True, help='resume model')
    parser.add_argument('--resume_model_path', type=str, default='/media/a/HDD/lyfeng/Face_Proj/FaceShifter/model/train2_003_256670.pth.tar', help='the model need to be loaded')
    parser.add_argument('--save_interval', type=int, default=2, help='how many epochs to save model')
    parser.add_argument('--save_path', type=str, default='', help='model save path')
    parser.add_argument('--save_dir', type=str, default='./model', help='model save dir')
    
    
    parser.add_argument('--gpus', type=str, default='2', help='model prefix')
    
    args = parser.parse_args()
    source, target, Yst0 = test(args)
    Yst0.save('yst0.jpg')
    source.save('source.jpg')
    target.save('taeget.jpg')
    
