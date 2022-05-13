import functools
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import time

from modules.cc_attention import CrissCrossAttention
from modules.dual_attention import CAM_Module, TAM_Module

class Non_Local_Attention(nn.Module):

    def __init__(self, nf=64, nframes=3):
        super(Non_Local_Attention, self).__init__()

        self.conv_before_cca = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                                   nn.ReLU())      
        self.conv_before_ca = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                                   nn.ReLU())
        self.conv_before_ta = nn.Sequential(nn.Conv2d(nframes, nframes, 3, padding=1, bias=False),
                                   nn.ReLU())

        self.recurrence = 2
        self.cca = CrissCrossAttention(nf)
        self.ca = CAM_Module()
        self.ta = TAM_Module()

        self.conv_after_cca = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                                   nn.ReLU())
        self.conv_after_ca = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                                   nn.ReLU())
        self.conv_after_ta = nn.Sequential(nn.Conv2d(nframes, nframes, 3, padding=1, bias=False),
                                   nn.ReLU())

        self.conv_final = nn.Conv2d(nf, nf, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  

        # spatial non-local attention
        cca_feat = self.conv_before_cca(aligned_fea.reshape(-1, C, H, W))
        for i in range(self.recurrence):
            cca_feat = self.cca(cca_feat)
        cca_conv = self.conv_after_cca(cca_feat).reshape(B, N, C, H, W)

        # channel non-local attention
        ca_feat = self.conv_before_ca(aligned_fea.reshape(-1, C, H, W))
        ca_feat = self.ca(ca_feat)
        ca_conv = self.conv_after_ca(ca_feat).reshape(B, N, C, H, W)

        # temporal non-local attention
        ta_feat = self.conv_before_ta(aligned_fea.permute(0, 2, 1, 3, 4).reshape(-1, N, H, W))
        ta_feat = self.ta(ta_feat)
        ta_conv = self.conv_after_ta(ta_feat).reshape(B, C, N, H, W).permute(0, 2, 1, 3, 4)

        feat_sum = cca_conv+ca_conv+ta_conv
        
        output = self.conv_final(feat_sum.reshape(-1, C, H, W)).reshape(B, N, C, H, W)
                
        return aligned_fea + output
    
if __name__ == "__main__":
#     a=Predenoiser()
#     print(sum(p.numel() for p in a.parameters())) #7698116
#     model = RViDeNet(a)
#     print(sum(p.numel() for p in model.parameters())) #8572916
    
    model = Non_Local_Attention(nf=64, nframes=1).cuda()
    print(sum(p.numel() for p in model.parameters())) #156981
    data = torch.ones((2,1,64,40,40)).cuda() # B T C H W
    print(model(data).shape)