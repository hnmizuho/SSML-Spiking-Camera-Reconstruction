import torch
import torch.nn as nn
from model import DoubleNet
import cv2
from utils import load_vidar_dat

if __name__ == '__main__':
    model = DoubleNet()
    model_path = "./fin3g-best-lucky.pt"
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path)) 
    model = model.cuda()
    
    spike_path = "./rotation1.dat"
    spike = load_vidar_dat(spike_path)[200:200+41].unsqueeze(0).cuda()

    res = model(spike)
    res = res[0].detach().cpu().permute(1,2,0).numpy()*255
    res_path = "./res.png"
    cv2.imwrite(res_path,res)

    print("done.")