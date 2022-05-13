import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

from mynonlocal import *

from cbam import SpatialGate,ChannelGate,Temporal_Fusion

class crop(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        x = x[0:N, 0:C, 0:H-1, 0:W]
        return x

class shift(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift_down = nn.ZeroPad2d((0,0,1,0))
        self.crop = crop()

    def forward(self, x):
        x = self.shift_down(x)
        x = self.crop(x)
        return x

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, blind=True,stride=1,padding=0,kernel_size=3):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift_down = nn.ZeroPad2d((0,0,1,0))
            self.crop = crop()
        self.replicate = nn.ReplicationPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding,bias=bias) 
        self.relu = nn.LeakyReLU(0.1, inplace=True)


    def forward(self, x):
        if self.blind:
            x = self.shift_down(x)
        x = self.replicate(x)
        x = self.conv(x)
        x = self.relu(x)        
        if self.blind:
            x = self.crop(x)
        return x

class Pool(nn.Module):
    def __init__(self, blind=True):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift = shift()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        if self.blind:
            x = self.shift(x)
        x = self.pool(x)
        return x

class rotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x90 = x.transpose(2,3).flip(3)
        x180 = x.flip(2).flip(3)
        x270 = x.transpose(2,3).flip(2)
        x = torch.cat((x,x90,x180,x270), dim=0)
        return x

class unrotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x0, x90, x180, x270 = torch.chunk(x, 4, dim=0)
        x90 = x90.transpose(2,3).flip(2)
        x180 = x180.flip(2).flip(3)
        x270 = x270.transpose(2,3).flip(3)
        x = torch.cat((x0,x90,x180,x270), dim=1)
        return x

class ENC_Conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias=False, reduce=True, blind=True):
        super().__init__()
        self.reduce = reduce
        self.conv1 = Conv(in_channels, mid_channels, bias=bias, blind=blind)
        self.conv2 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv3 = Conv(mid_channels, out_channels, bias=bias, blind=blind)
        if reduce:
            self.pool = Pool(blind=blind)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.reduce:
            x = self.pool(x)
        return x

class DEC_Conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias=False, blind=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = Conv(in_channels, mid_channels, bias=bias, blind=blind)
        self.conv2 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv3 = Conv(mid_channels, mid_channels, bias=bias, blind=blind)
        self.conv4 = Conv(mid_channels, out_channels, bias=bias, blind=blind)

    def forward(self, x, x_in):
        x = self.upsample(x)

        # Smart Padding
        diffY = x_in.size()[2] - x.size()[2]
        diffX = x_in.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        x = torch.cat((x, x_in), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class Blind_UNet(nn.Module):
    def __init__(self, n_channels=3, n_output=96, bias=False, blind=True):
        super().__init__()
        self.n_channels = n_channels
        self.bias = bias
        self.enc1 = ENC_Conv(n_channels, 48, 48, bias=bias, blind=blind)
        self.enc2 = ENC_Conv(48, 48, 48, bias=bias, blind=blind)
        self.enc3 = ENC_Conv(48, 96, 48, bias=bias, reduce=False, blind=blind)
        self.dec2 = DEC_Conv(96, 96, 96, bias=bias, blind=blind)
        self.dec1 = DEC_Conv(96+n_channels, 96, n_output, bias=bias, blind=blind)

    def forward(self, input):
        x1 = self.enc1(input)
        x2 = self.enc2(x1)
        x = self.enc3(x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, input)
        return x

class BlindVideoNetSpike(nn.Module): #omg21
    def __init__(self, channels_per_frame=3, out_channels=9, bias=False, blind=True, sigma_known=True,args=None):
        super().__init__()
        self.c = channels_per_frame
        self.out_channels = out_channels
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        self.denoiser_1 = Blind_UNet(n_channels=43*channels_per_frame, n_output=32, bias=bias, blind=blind)
        self.denoiser_2 = Blind_UNet(n_channels=96, n_output=96, bias=bias, blind=blind)
        if blind:
            self.shift = shift()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, out_channels, 1, bias=bias)
        self.non_local = Non_Local_Attention(nf=32, nframes=3)
        self.MotionInference = MotionInference(n_frame=27,bias=bias,blind=blind)

    def forward(self, x):
        # Square
        N, C, H, W = x.shape
        _,tfi_label,tfp_label = self.MotionInference(x[:, 0:27, :, :])
        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')

        i1 = self.rotate(x[:, 0:27, :, :]) #16 3 64 64
        i2 = self.rotate(x[:, 7:34, :, :])
        i3 = self.rotate(x[:, 14:41, :, :])

        fea1,_,_ = self.MotionInference(i1)
        fea2,_,_ = self.MotionInference(i2)
        fea3,_,_ = self.MotionInference(i3) #b 16 h w
        
        i1 = torch.cat([i1,fea1],dim=1) #16 3 64 64
        i2 = torch.cat([i2,fea2],dim=1)
        i3 = torch.cat([i3,fea3],dim=1)

        #stage1
        y1 = self.denoiser_1(i1) #16 32 64 64 
        y2 = self.denoiser_1(i2)
        y3 = self.denoiser_1(i3)

        #stage2
#         y = torch.cat((y1, y2, y3), dim=1) #fin1
        ffea = torch.stack([y1,y2,y3],dim=1) #fin2
        nonlocal_fusion = self.non_local(ffea)   
        bb,tt,cc,hh,ww = nonlocal_fusion.shape
        y = nonlocal_fusion.view(bb,-1,hh,ww)
        
        x = self.denoiser_2(y) #16 96 64 64

        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x) #4 384 64 64
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)
        
        if(H > W):
            diff = H - W
            x = x[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x = x[:, :, (diff // 2):(diff // 2 + H), 0:W]

        return x,tfi_label,tfp_label

class MotionInference(nn.Module):
    def __init__(self,n_frame=41,bias=False,blind=False):
        super().__init__()
        self.middle = n_frame//2
        self.conv0 = nn.Conv2d(5*2+1,1,1,bias=bias)
        self.conv1 = nn.Conv2d(9*2+1,1,1,bias=bias)
        self.conv2 = nn.Conv2d(13*2+1,1,1,bias=bias)
        self.tfpconv = Conv(in_channels=3, out_channels=16, bias=bias,blind=blind)
        self.tficonv = Conv(in_channels=1, out_channels=16, bias=bias,blind=blind)
        self.ChannelGate = ChannelGate(gate_channels=16, reduction_ratio=4)
        self.SpatialGate = SpatialGate(bias=bias,blind=blind)
        self.blind = blind
    def forward(self, x):  
        N, C, H, W = x.shape
        tmp=[]
        ttt=[]
        for j in range(N):
            tmp2 = middleTFI(x[j].cpu(), self.middle, window=12)    
            tmp2 = torch.tensor(tmp2,dtype=torch.float32).unsqueeze_(dim=0)
            tmp.append(tmp2) #1 40 40
            ttt5=torch.mean(x[j,self.middle-3:self.middle+3+1,:,:].cpu(),dim=0).unsqueeze_(0) 
            ttt.append(ttt5)
        tfi_label = torch.stack(tmp,0).cuda()
        tfp_label = torch.stack(ttt,0).cuda()

#         tfi_label = (torch.clamp(tfi_label,0,1))**(1/2.2)
#         tfp_label = (torch.clamp(tfp_label,0,1))**(1/2.2)
        
        tfp0 = self.conv0(x[:,self.middle-5:self.middle+5+1,:,:]) #b 1 h w，
        tfp1 = self.conv1(x[:,self.middle-9:self.middle+9+1,:,:])
        tfp2 = self.conv2(x[:,self.middle-13:self.middle+13+1,:,:])
        tfps = torch.cat([tfp0,tfp1,tfp2],dim=1) #b 3 h w
        
        tfp_fea = self.tfpconv(tfps)        
        tfi_fea = self.tficonv(tfi_label)
        
        if not self.blind:
            tfp_fea = self.SpatialGate(tfp_fea) #b 16 h w
            tfi_fea = self.SpatialGate(tfi_fea) 
            fusion_fea = self.ChannelGate(tfp_fea+tfi_fea) #b 16 h w
        else:
            fusion_fea = tfp_fea+tfi_fea
#         tfi_label = (torch.clamp(tfi_label,0,1))**(1/2.2)
#         tfp_label = (torch.clamp(tfp_label,0,1))**(1/2.2)        
        return fusion_fea,tfi_label,tfp_label
        
class NorotaMultiSpike(nn.Module): 
    def __init__(self, n_channels=3, n_output=3, bias=False, blind=False,args=None):
        super().__init__()
        self.n_channels = n_channels
        self.c = n_channels
        self.n_output = n_output
        self.bias = bias
        self.blind = blind
        self.denoiser_1 = Blind_UNet(n_channels=43, n_output=32, bias=bias, blind=blind) #43-->16
        self.denoiser_2 = Blind_UNet(n_channels=32*3, n_output=96*4, bias=bias, blind=blind) #96+3-->3+1
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, n_output, 1, bias=bias)
        # self.non_local = Non_Local_Attention(nf=32, nframes=3) 
        self.MotionInference = MotionInference(n_frame=27,bias=bias,blind=blind) 
        
    def forward(self, x):  
        N, C, H, W = x.shape 
        # 13 20 27
        fea1,_,_ = self.MotionInference(x[:, 0:27, :, :])
        fea2,tfi_label,tfp_label = self.MotionInference(x[:, 7:34, :, :])
        fea3,_,_ = self.MotionInference(x[:, 14:41, :, :]) #b 16 h w
        
        i1 = torch.cat([x[:, 0:27, :, :],fea1],dim=1) #4 27+16 40 40 
        i2 = torch.cat([x[:, 7:34, :, :],fea2],dim=1)
        i3 = torch.cat([x[:, 14:41, :, :],fea3],dim=1)


        y1 = self.denoiser_1(i1)
        y2 = self.denoiser_1(i2)
        y3 = self.denoiser_1(i3) #b 32 h w

        # ffea = torch.stack([y1,y2,y3],dim=1) #b 3 32 h w omg48
        # nonlocal_fusion = self.non_local(ffea)   
        # bb,tt,cc,hh,ww = nonlocal_fusion.shape
        # y = nonlocal_fusion.view(bb,-1,hh,ww)
        y = torch.cat([y1,y2,y3],dim=1)

        x = self.denoiser_2(y) #4 96*4 64 64
        x = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x = F.leaky_relu_(self.nin_B(x), negative_slope=0.1)
        x = self.nin_C(x)
            
        return x,tfi_label,tfp_label

class BSN(nn.Module): 
    def __init__(self, n_channels=3, n_output=3, bias=False, blind=True, sigma_known=True):
        super().__init__()
        self.n_channels = n_channels
        self.c = n_channels
        self.n_output = n_output
        self.bias = bias
        self.blind = blind
        self.sigma_known = sigma_known
        self.rotate = rotate()
        self.unet = Blind_UNet(n_channels=n_channels+16, bias=bias, blind=blind) 
        self.shift = shift()
        self.unrotate = unrotate()
        self.nin_A = nn.Conv2d(384, 384, 1, bias=bias)
        self.nin_B = nn.Conv2d(384, 96, 1, bias=bias)
        self.nin_C = nn.Conv2d(96, n_output, 1, bias=bias)
        self.MotionInference = MotionInference(n_frame=41,bias=bias,blind=blind)
    
    def forward(self, x):  
        N, C, H, W = x.shape   
        _,tfi_label,tfp_label = self.MotionInference(x) 
        if(H > W):
            diff = H - W
            x = F.pad(x, [diff // 2, diff - diff // 2, 0, 0], mode = 'reflect')
        elif(W > H):
            diff = W - H
            x = F.pad(x, [0, 0, diff // 2, diff - diff // 2], mode = 'reflect')
            
        x = self.rotate(x)
        
        fea1,tfi,tfp = self.MotionInference(x)
        x = torch.cat([x,fea1],1)
        
        x = self.unet(x)   #4 3 100 100 -> 4 96 100 100
        if self.blind:
            x = self.shift(x)
        x = self.unrotate(x) #4 96 100 100 -> 1 384 100 100
        
        x0 = F.leaky_relu_(self.nin_A(x), negative_slope=0.1)
        x0 = F.leaky_relu_(self.nin_B(x0), negative_slope=0.1)
        x0 = self.nin_C(x0)

        # Unsquare
        if(H > W):
            diff = H - W
            x0 = x0[:, :, 0:H, (diff // 2):(diff // 2 + W)]
        elif(W > H):
            diff = W - H
            x0 = x0[:, :, (diff // 2):(diff // 2 + H), 0:W]
            
        return x0,tfi_label,tfp_label   

class DoubleNet(nn.Module):
    def __init__(self,args=None):
        super().__init__()
        self.nbsn = NorotaMultiSpike(n_channels=41, n_output=1,blind=False,args=args)
        # self.nbsn = BSN(n_channels=41, n_output=1,blind=False)
#         self.bsn = BlindVideoNetSpike(channels_per_frame=1, out_channels=1,blind=True,args=args)
        self.bsn = BSN(n_channels=41, n_output=1,blind=True)
    
    def forward(self, x):  
        out1,tfi,tfp = self.nbsn(x)
        out2=out1
#         out2,_,_= self.bsn(x) #生成视频时不要这一句，快五六倍
        
        return out1,out2,tfi,tfp

if __name__ == '__main__':
    a=Conv(1,4,kernel_size=7,padding=2,stride=1,blind=False)
    print(a(torch.ones(2,1,40,40)).shape)
