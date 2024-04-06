import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import set_matrix, pose2mat
from net.GlobalPoolformer import GlobalPoolformerEncoder
from net.INN import InvertibleNN
from net.FFC import FFCResnetBlock
import ProSTGrid
from net.ProST import raydist_range,_bilinear_interpolate_no_torch_5D,net3D
from einops import rearrange
class CD_net(nn.Module):
    def __init__(self):
        super(CD_net, self).__init__()
        self.shallowFeatureEncoder = GlobalPoolformerEncoder(img_size=((128,128),(64,64),(32,32), (16,16)), in_channels=3, features=(8,16,32,64,64),
                                                             pool_size=((2, 2),(2, 2), (2, 2),(2,2)))
        self.encoder_l_x = FFCResnetBlock(64, padding_type='reflect', norm_layer=nn.BatchNorm2d,inline=True)
        self.encoder_l_d = FFCResnetBlock(64, padding_type='reflect', norm_layer=nn.BatchNorm2d,inline=True)
        self.encoder_h_x = InvertibleNN(num_layers=3)
        self.encoder_h_d = InvertibleNN(num_layers=3)
        self.detailEstimation =MLP()
        self.baseEstimation=MLP()
        self._3D_conv = nn.Sequential(
            nn.Conv3d(1, 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(4, 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(16, 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(8, 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(4, 3, 3, 1, 1),
            nn.ReLU()
        )
    def forward(self, x, y, corner_pt, param, rtvec):
        BATCH_SIZE = rtvec.size()[0]
        transform_mat3x4 = set_matrix(BATCH_SIZE, 'cuda', rtvec)
        # x is ct, y is x-ray
        src = param[0]
        det = param[1]
        pix_spacing = param[2]
        step_size = param[3]
        x_exp = x.repeat(1, 3, 1, 1, 1)
        x_3d = x_exp + self._3D_conv(x)  # Involve residential part
        H = y.size()[2]
        W = y.size()[3]

        dist_min, dist_max = raydist_range(transform_mat3x4, corner_pt, src)
        grid = ProSTGrid.forward(corner_pt, y.size(), dist_min.data, dist_max.data,
                                 src, det, pix_spacing, step_size, False)
        grid_trans = grid.bmm(transform_mat3x4.transpose(1, 2)).view(BATCH_SIZE, H, W, -1, 3)
        x_3d = _bilinear_interpolate_no_torch_5D(x_3d, grid_trans)
        drr = torch.sum(x_3d, dim=-1)
        y = y.repeat(1, 3, 1, 1)
        SF_x = self.shallowFeatureEncoder(y)
        SF_d = self.shallowFeatureEncoder(drr)
        L_x = self.encoder_l_x(SF_x)
        H_x = self.encoder_h_x(SF_x)
        L_d = self.encoder_l_d(SF_d)
        H_d = self.encoder_h_d(SF_d)
        
        H = H_d - H_x
        H2=H*H
        L=L_d-L_x
        L2=L*L
        H2=rearrange(H2, 'b c h w -> b (c h w)')
        L2=rearrange(L2, 'b c h w -> b (c h w)')
        H3,he=self.detailEstimation(H2)
        L3,le=self.baseEstimation(L2)
        return L_x, H_x, L_d, H_d,H3,L3,he,le
        # return drr

class MLP(nn.Module):
    
     def __init__(self):
        super(MLP, self).__init__()
        self.fc1=nn.Linear(16384,8192)
        self.fc2=nn.Linear(8192,4096)
        self.fc3=nn.Linear(4096,1024)
        self.fc4=nn.Linear(1024,256)
        self.fc5=nn.Linear(256,1)
        
        self.fc6=nn.Linear(4096,1024)
        self.fc7=nn.Linear(1024,256)
        self.fc8=nn.Linear(256,1)
     def forward(self,x):
        x=F.relu(self.fc1(x))
        x1=F.relu(self.fc2(x))
        
        x2=F.relu(self.fc6(x1))
        x2=F.relu(self.fc7(x2))
        x2=F.sigmoid(self.fc8(x2))
        
        x=F.relu(self.fc3(x1))
        x=F.relu(self.fc4(x))
        x=self.fc5(x)
        
        
       
        
        out=x*x2
        return out,x2
    
    
class UncertaintyLoss(nn.Module):

    def __init__(self, v_num):
        super(UncertaintyLoss, self).__init__()
        sigma = torch.randn(v_num)
        self.sigma = nn.Parameter(sigma)
        self.v_num = v_num

    def forward(self, *input):
        loss = 0
        for i in range(self.v_num):
            loss =loss+ input[i] / (2 * self.sigma[i] ** 2)
        loss = loss+ torch.log(self.sigma.pow(2).prod())
        return loss
