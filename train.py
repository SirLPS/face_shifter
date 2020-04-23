# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:04:43 2020
a new version of training loss
@author: lenovo

"""
import torch
from torch.nn import DataParallel
from network import Att_Encoder, AAD_Gen
from get_feats import ArcFace_Net
from dataloader import SupplyCollate
from metric import loss_hinge_dis, loss_hinge_gen, IdLoss, AttrLoss, RecLoss
from metric import loss_hinge_dis_mse, loss_hinge_gen_mse
from utils import MultiscaleDiscriminator
import torchvision.transforms as trans
import torch.nn as nn
import argparse
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import time
from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import interpolate as downsample
from model import Backbone
from dataloader import FaceEmbed


def train(args):
    # gpu init
    multi_gpu = False
    if len(args.gpus.split(',')) > 1:
        multi_gpu = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    D = MultiscaleDiscriminator(input_nc=3, ndf=64, n_layers=3, use_sigmoid=False,norm_layer=torch.nn.InstanceNorm2d)    # pix2pix use MSEloss
    G = AAD_Gen()
    F = Backbone(50, drop_ratio=0.6, mode='ir_se')
    F.load_state_dict(torch.load( args.arc_model_path))   
    E = Att_Encoder()
     
    optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0004, betas=(0.0, 0.999))
    optimizer_GE = torch.optim.Adam([{'params': G.parameters()}, 
                                   {'params': E.parameters()}],
                                   lr=0.0004, betas=(0.0, 0.999))
    
    
    if multi_gpu:
        D = DataParallel(D).to(device)
        G = DataParallel(G).to(device)
        F = DataParallel(F).to(device)
        E = DataParallel(E).to(device)
    else:
        D = D.to(device)
        G = G.to(device)
        F = F.to(device)
        E = E.to(device)
    
    if args.resume:
        if os.path.isfile(args.resume_model_path):
            print("Loading checkpoint from {}". format(args.resume_model_path))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            D.load_state_dict(checkpoint["state_dict_D"])
            G.load_state_dict(checkpoint["state_dict_G"])
            E.load_state_dict(checkpoint["state_dict_E"])
#            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            optimizer_GE.load_state_dict(checkpoint['optimizer_GE'])
        else:
            print('Cannot found checkpoint {}'.format(args.resume_model_path))
    else:
        args.start_epoch = 1
        
           
    def print_with_time(string):
        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + string)
        
    def weights_init(m):
        classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)        
            
    def set_requires_grad( nets, requires_grad=False):

        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def trans_batch(batch):
            t =  trans.Compose([trans.ToPILImage(), trans.Resize((112,112)), trans.ToTensor()])   
            bs = batch.shape[0]
            res = torch.ones(bs,3,112,112).type_as(batch)
            for i in range(bs):
                res[i] = t(batch[i].cpu())
            return res  
             

    set_requires_grad(F, requires_grad=False)
    data_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    #dataset = ImageFolder(args.data_path, transform=data_transform)  
    dataset = FaceEmbed(args.data_path)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    D.apply(weights_init)
    G.apply(weights_init)
    E.apply(weights_init)

    for epoch in range(args.start_epoch, args.total_epoch+1):    
        D.train()
        G.train()
        F.eval()    #   Only extract features!  # input dim=3,256,256   out dim=256 ! 
        E.train()

        for batch_idx, data in enumerate(data_loader):
            time_curr = time.time()
            iteration = (epoch - 1) * len(data_loader) + batch_idx
            try:
                source,target, label = data
            
                source = source.to(device)
                target =target.to(device)
                label = torch.LongTensor(label).to(device)
                   
                #Zid =F(trans_batch(source))  # bs, 512
                Zid = F(downsample(source[:,:,50:-10, 30:-30] , size=(112,112)))
                Zatt = E(target)     # list:8  each：bs,,,
                Yst0 = G(Zid, Zatt)  # bs,3,256,256
            
                # train discriminators
                pred_gen = D(Yst0.detach())
                #pred_gen = list(map(lambda x: x[0].detach(), pred_gen))
                pred_real = D(target)
                optimizer_D.zero_grad() 
                loss_real, loss_fake = loss_hinge_dis()(pred_gen, pred_real)
                L_dis = loss_real + loss_fake
            #    if batch_idx%3==0:
                L_dis.backward()
                optimizer_D.step()
                    
                    
                # train generators              
                pred_gen = D(Yst0)
                L_gen = loss_hinge_gen()(pred_gen)      
                #L_id = IdLoss()(F(trans_batch(Yst0)), Zid)
                L_id = IdLoss()(F(downsample(Yst0[:,:,50:-10, 30:-30] , size=(112,112))), Zid)
                #Zatt = list(map(lambda x: x.detach(), Zatt))
                L_att = AttrLoss()( E(Yst0), Zatt)
                L_Rec = RecLoss()(Yst0, target, label)
                
                Loss = (L_gen + 10*L_att + 5*L_id + 10*L_Rec).to(device)
                optimizer_GE.zero_grad() 
                Loss.backward()
                optimizer_GE.step()
                
            except Exception as e:
                print(e)
                continue

            if batch_idx % args.log_interval == 0 or batch_idx==20:
                time_used = time.time() - time_curr
                print_with_time(
                    'Train Epoch: {} [{}/{} ({:.0f}%)], L_dis:{:.4f}, loss_real:{:.4f}, loss_fake:{:.4f}, Loss:{:.4f}, L_gen:{:.4f}, L_id:{:.4f}, L_att:{:.4f}, L_Rec:{:.4f}'.format(
                        epoch, batch_idx * len(data), len(data_loader.dataset), 100. * batch_idx *len(data)/ len(data_loader.dataset),
                        L_dis.item(), loss_real.item(), loss_fake.item(), Loss.item(), L_gen.item(), 5*L_id.item(),10* L_att.item(), 10*L_Rec)
                )
                time_curr = time.time()
            
        if epoch % args.save_interval == 0: #or batch_idx*len(data) % 350004==0:
            state = {
                "epoch": epoch,
                "state_dict_D": D.state_dict(),
                "state_dict_G": G.state_dict(),
                "state_dict_E": E.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
                "optimizer_GE": optimizer_GE.state_dict(),
#                        "optimizer_E": optimizer_E.state_dict(),
            }
            filename = "../model/train1_{:03d}_{:03d}.pth.tar".format(epoch, batch_idx*len(data))
            torch.save(state, filename)
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch for FaceShifter')
    parser.add_argument('--backbone', type=str, default='resnet50', help='resnet18, resnet50, resnet101, resnet152')
    parser.add_argument('--arc_model_path', type=str, default='/media/a/HDD/Face_Proj/FaceShifter/src/model_ir_se50.pth', help='path to arcface pretrained model')

    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension, 256 or 512. original is 256 !!!')
    parser.add_argument('--data_path', type=str, default = '/media/a/HDD/Face_Proj/vgg_face_dataset/new_images')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')
    parser.add_argument('--start_epoch', type=int, default=1, help= 'the start of epoch')
    parser.add_argument('--total_epoch', type=int, default=50, help='total epochs')
    parser.add_argument('--dis_times', type=int, default=1, help='how often update discriminators ')
    parser.add_argument('--gen_times', type=int, default=1, help='how often update generators ')

    parser.add_argument('--log_interval', type=int, default=500, help='how many batches to wait before logging training status')
    parser.add_argument('--lr_step', type=int, default=10, help= 'lr decay step')
    parser.add_argument('--resume', type=int, default=False, help='resume model')
    parser.add_argument('--resume_model_path', type=str, default='', help='the model need to be loaded')
    parser.add_argument('--save_interval', type=int, default=1, help='how many epochs to save model')
    parser.add_argument('--save_path', type=str, default='', help='model save path')
    parser.add_argument('--save_dir', type=str, default='./model', help='model save dir')
    
    
    parser.add_argument('--gpus', type=str, default='0', help='model prefix')
    
    args = parser.parse_args()
    train(args)
    

