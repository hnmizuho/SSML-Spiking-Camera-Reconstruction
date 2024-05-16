from __future__ import print_function
import argparse
from datetime import datetime
import json
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import Adam, lr_scheduler
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import cv2
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from model import BSN,DoubleNet
from utils import load_vidar_dat,plot_per_epoch, progress_bar, time_elapsed_since, middleTFI

class DatasetOneSpike(Dataset):
    def __init__(self, root_dir, crop_size, spike_len):
        super(DatasetOneSpike, self).__init__()
        self.imgfolder = root_dir+"/input/"
        
        self.spike_len = spike_len
        self.img_list = os.listdir(self.imgfolder)
        self.long_spike = load_vidar_dat(os.path.join(self.imgfolder,self.img_list[0]))
        
        self.crop_size = crop_size

        self.transform = transforms.Compose([
            transforms.RandomCrop(self.crop_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            ])
    
    def normalize(self, tfp):
        tfp = (tfp - torch.min(tfp)) / (torch.max(tfp) - torch.min(tfp) + 1e-8)
        return tfp

    def __getitem__(self, index: int):
        spike = self.long_spike[index:index+self.spike_len]

        if self.crop_size != 0:
            spike = self.transform(spike) 
        
        tfp = torch.mean(spike[len(spike)//2-3:len(spike)//2+3+1],dim=0,keepdim=True)
        tfp = self.normalize(tfp)
        tfp = tfp**(1/2.2)

        item = {}
        item['spike'] = spike
        item['tfp'] = tfp

        return item

    def __len__(self) -> int:
        return self.long_spike.shape[0] - self.spike_len + 1

def save_network_bsn(network, save_path):
    if isinstance(network, nn.DataParallel):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)

# def save_network_nbsn(network, save_path):
#     if isinstance(network, nn.DataParallel):
#         network = network.module
#     state_dict = network.nbsn.state_dict()
#     for key, param in state_dict.items():
#         state_dict[key] = param.cpu()
#     torch.save(state_dict, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default='./data', help='path to dataset')
    parser.add_argument('--valid-dir', default='./data', help='path to dataset')

    parser.add_argument('--ckpt-dir', default='./ckpts/expr1', help='folder to output images and model checkpoints')

    parser.add_argument('--batch-size', type=int, default=8, help='input batch size')
    parser.add_argument('--crop-size', type=int, default=128, help='the height / width of the input image to network')
    parser.add_argument('--spike-len', type=int, default=41, help='spike len')

    parser.add_argument('--nb_epochs', type=int, default=5, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate, default=0.0002')

    parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')

    opt = parser.parse_args()
    print(opt)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    
    crop_size = opt.crop_size
    train_dir = opt.train_dir
    valid_dir = opt.valid_dir
    batch_size = opt.batch_size
    spike_len = opt.spike_len
    nb_epochs = opt.nb_epochs
    ckpt_dir = opt.ckpt_dir
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    train_dataset = DatasetOneSpike(train_dir,crop_size,spike_len)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=20,pin_memory=True)

    valid_dataset = DatasetOneSpike(valid_dir,0,spike_len)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False,num_workers=20,pin_memory=True)

    # model = BSN(n_channels=spike_len, n_output=1, bias=False, blind=True)
    model = DoubleNet(spike_len, 1)
    optim = Adam(model.parameters(), lr=opt.lr,betas=[0.9,0.99])
    scheduler = lr_scheduler.MultiStepLR(optim, milestones=[1,2,3,4,5], gamma=0.5)   
    criterion = nn.MSELoss()

    model = model.cuda()
    model = nn.DataParallel(model)

    max_psnr = 0

    print("Start training!")
    train_start = datetime.now()
    stats = {'PSNR':[]}
    for epoch in range(nb_epochs):
        print('EPOCH {:d} / {:d}'.format(epoch + 1, nb_epochs))
        epoch_start = datetime.now()
        
        loss = 0
        model.train(True)
        for batch_idx, item in enumerate(train_loader):
            progress_bar(batch_idx, len(train_loader), 50, loss)

            source = item['spike'].cuda()
            tfp = item['tfp'].cuda()

            optim.zero_grad()

            # bsn_pred = model(source)
            # loss1 = criterion(bsn_pred,tfp)

            bsn_pred, nbsn_pred = model(source,test=False)
            loss1 = criterion(bsn_pred,tfp) + 0.01*criterion(nbsn_pred,bsn_pred)

            loss = loss1 + 0
            
            loss.backward()
            optim.step()
            # break

        scheduler.step()

        ####### eval #######
        if epoch % 1 == 0:
            model.eval()
            psnr = 0
            # simple eval. This PSNR is meaningless without GT.
            for batch_idx, item in enumerate(valid_loader):
                source = item['spike'].cuda()
                tfp = item['tfp'].cuda()
                optim.zero_grad()
                
                with torch.no_grad():
                    nbsn_pred = model(source)
                    psnr += compare_psnr(tfp[0].cpu().detach().numpy(),nbsn_pred[0].cpu().detach().numpy(),data_range=1)
                break
            psnr /= 1
            # psnr /= len(valid_dataset)
            print("PSNR: {:.4f}".format(psnr))
            stats['PSNR'].append(psnr)

            path = "{}/ckpt_{}.pth".format(ckpt_dir, epoch)
            save_network_bsn(model, path)
            fname_dict = '{}/stats.json'.format(ckpt_dir)
            with open(fname_dict, 'w') as fp:
                json.dump(stats, fp, indent=2)
            plot_per_epoch(ckpt_dir, 'Valid PSNR', stats['PSNR'], 'PSNR (dB)')

            # save nbsn_pred image
            nbsn_pred = nbsn_pred[0].permute(1,2,0).cpu().detach().numpy()
            cv2.imwrite("{}/nbsn_pred_{}.png".format(ckpt_dir,epoch),nbsn_pred*255)
            tfp = tfp[0].permute(1,2,0).cpu().detach().numpy()
            cv2.imwrite("{}/tfp_{}.png".format(ckpt_dir,epoch),tfp*255)

        epoch_time = time_elapsed_since(epoch_start)[0]
        print('epoch_time: {}\n'.format(epoch_time))

    train_elapsed = time_elapsed_since(train_start)[0]
    print('Training done! Total elapsed time: {}\n'.format(train_elapsed))