# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:39:53 2020

@author: lenovo
"""

import torch.nn.functional as F
from utils import *
import torch 



class Att_Encoder(nn.Module):

    def __init__(self):
        super(Att_Encoder,self).__init__()
        self.bilinear = BilinearUp()
        
        self.down1 = Conv(3, 32)       # Joint Discriminative and Generative Learning for Person ReID
        self.down2 = Conv(32, 64)
        self.down3 = Conv(64, 128)
        self.down4 = Conv(128, 256)
        self.down5 = Conv(256, 512)
        self.down6 = Conv(512, 1024)
        self.down7 = Conv(1024, 1024)
        
        self.up1 = ConvTrans(1024, 1024)
        self.up2 = ConvTrans(2048, 512)
        self.up3 = ConvTrans(1024, 256)
        self.up4 = ConvTrans(512, 128)      
        self.up5 = ConvTrans(256, 64)
        self.up6 = ConvTrans(128, 32)     
        
    def forward(self, x):
        """
        avoid  a trivial solution for G
        
        """
#        x = x[:,0,:,:]*0.299+x[:,1,:,:]*0.587+x[:,2,:,:]*0.114
#        x = x.unsqueeze(1)
        x1 = self.down1(x)   # 32,128,128
        x2 = self.down2(x1)  # 64,64,64
        x3 = self.down3(x2)  # 128,32,32
        x4 = self.down4(x3)  # 256,16,16
        x5 = self.down5(x4)  # 512,8,8
        x6 = self.down6(x5)  # 1024,4,4 
        x7 = self.down7(x6)  # 1024,2,2
        
#        y1 = x7
        y2 = self.up1(x7)                   # 1024,4,4
        y3 = self.up2(torch.cat((y2, x6), dim=1))    # in:2048,4,4    out: 512,8,8
        y4 = self.up3(torch.cat((y3, x5), dim=1))    # in:1024,8,8    out: 256,16,16
        y5 = self.up4(torch.cat((y4, x4), dim=1))    # in:512,16,16   out: 128,32,32
        y6 = self.up5(torch.cat((y5, x3), dim=1))    # in:256,32,32   out: 64,64,64
        y7 = self.up6(torch.cat((y6, x2), dim=1))    # in:128,64,64   out: 32,128,128
        y8 = self.bilinear(torch.cat((y7,x1), dim=1))       # in: 64,128,128   out: 64,256,256
        
#        Zatt1 = y1                          #  1024,2,2
        Zatt2 = torch.cat((y2, x6), dim=1)   #  2048,4,4
        Zatt3 = torch.cat((y3, x5), dim=1)   #  1024,8,8
        Zatt4 = torch.cat((y4, x4), dim=1)   #  512,16,16
        Zatt5 = torch.cat((y5, x3), dim=1)   #  256,32,32
        Zatt6 = torch.cat((y6, x2), dim=1)   #  128,64,64
        Zatt7 = torch.cat((y7, x1), dim=1)   #  64,128,128
#        Zatt8 = y8                          #  64,256,256 
        
        
        return [x7, Zatt2, Zatt3, Zatt4, Zatt5, Zatt6, Zatt7, y8]

    
    
class AAD_Gen(nn.Module):
    """
    generator network
    """
    def __init__(self):
        super(AAD_Gen,self).__init__()
        # each block: in_dim, out_dim, att_dim
        self.block1 = AAD_ResBlk(1024, 1024, 1024)   
        self.block2 = AAD_ResBlk(1024, 1024, 2048)
        self.block3 = AAD_ResBlk(1024, 1024, 1024)
        self.block4 = AAD_ResBlk(1024, 512, 512)
        self.block5 = AAD_ResBlk(512, 256, 256)
        self.block6 = AAD_ResBlk(256, 128, 128)
        self.block7 = AAD_ResBlk(128, 64, 64)
        self.block8 = AAD_ResBlk(64, 3, 64)

        self.bilinear = BilinearUp()
        self.ConvTrans = nn.ConvTranspose2d(
                512, 1024, kernel_size=2, stride=1, padding=0    # dim in paper is 256 !!!
                )

    def forward(self, Z_id, Zatt):
        Z_id  = Z_id.unsqueeze(-1).unsqueeze(-1)
        x = self.block1(self.ConvTrans(Z_id), Zatt[0], Z_id)
        x = self.block2(self.bilinear(x), Zatt[1], Z_id)
        x = self.block3(self.bilinear(x), Zatt[2], Z_id)
        x = self.block4(self.bilinear(x), Zatt[3], Z_id)
        x = self.block5(self.bilinear(x), Zatt[4], Z_id)
        x = self.block6(self.bilinear(x), Zatt[5], Z_id)
        x = self.block7(self.bilinear(x), Zatt[6], Z_id)
        x = self.block8(self.bilinear(x), Zatt[7], Z_id)

        return x   #  Y^_st
    

class HEARNet(nn.Module):
    """
    hear network
    """
    def __init__(self):
        super().__init__()
        self.bilinear = BilinearUp()
        
        self.down1 = Conv(6, 64)
        self.down2 = Conv(64, 128)
        self.down3 = Conv(128, 256)
        self.down4 = Conv(256, 512)
        self.down5 = Conv(512, 512)
        
        self.up1 = ConvTrans(512, 512)
        self.up2 = ConvTrans(1024, 256)
        self.up3 = ConvTrans(512, 128)
        self.up4 = ConvTrans(256, 64)      
        self.up5 = ConvTrans(128, 3)  
        
    def forward(self, x):
        """ x should be 6,256,256
            concat((3,256,256),(3,256,256))
        """
        x1 = self.down1(x)   # 64,128,128
        x2 = self.down2(x1)  # 128,64,64
        x3 = self.down3(x2)  # 256,32,32
        x4 = self.down4(x3)  # 512,16,16
        x5 = self.down5(x4)  # 512,8,8
        
        y1 = self.up1(x5)    # 512,16,16
        y2 = self.up2(torch.cat((x4, y1), dim=1))    # 256,32,32
        y3 = self.up3(torch.cat((x3, y2), dim=1))    # 128,64,64
        y4 = self.up4(torch.cat((x2, y3), dim=1))    # 64,128,128
        y5 = self.up5(torch.cat((x1, y4), dim=1))    # 64,64,64
        
        return y5
