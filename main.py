#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from utils import *

import os
import json
import copy
from tqdm import tqdm

from torch.optim.lr_scheduler import _LRScheduler
from mynet import *
from torch.utils.tensorboard import SummaryWriter

class Trainer(object):

    def __init__(self, params, trainable):
        """Initializes model."""

        self.p = params
        self.trainable = trainable
        self._compile()
        self.gamma = 2 


    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

        print('Compiling model...')

        if self.p.net == "csy":
            if self.p.subtask == "double":
                self.model = DoubleNet(args=self.p)
            
        weights_init_kaiming(self.model)
        
        
        # Set optimizer and loss, if in training mode
        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=self.p.learning_rate,
                              betas=self.p.adam[:2],
                              eps=self.p.adam[2])

            if self.p.scheduler == "plateau":
                self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, patience=8, factor=0.5, verbose=True)
            elif self.p.scheduler == "cos":
                if self.p.bigdataset == 0:
                    self.scheduler = lr_scheduler.CosineAnnealingLR(self.optim, T_max=self.p.nb_epochs) 
                if self.p.bigdataset != 0:
                    self.scheduler = lr_scheduler.CosineAnnealingLR(self.optim, T_max=1000) 
            elif self.p.scheduler == "step":
                self.scheduler = lr_scheduler.StepLR(self.optim, step_size=10, gamma=0.5)     
            
            self.distill_loss = torch.nn.MSELoss().cuda()
            if self.p.loss == 'l2':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()

        # CUDA support
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        if self.use_cuda:
            self.model = self.model.cuda()
            self.model = nn.DataParallel(self.model)
            if self.trainable:
                self.loss = self.loss.cuda()
        #temnsorboard
        self.writer = SummaryWriter(log_dir = os.path.join(self.p.ckpt_save_path, self.p.exper_id,self.p.exper_id))

    def _print_params(self):
        """Formats parameters to print when training."""

        print('Training parameters: ')
        self.p.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()


    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            if self.p.clean_targets:
                ckpt_dir_name = f'{datetime.now():{self.p.exper_id}-clean-%H%M}'
            else:
                ckpt_dir_name = f'{datetime.now():{self.p.exper_id}-%H%M}'
            if self.p.ckpt_overwrite:
                if self.p.clean_targets:
                    ckpt_dir_name = f'{self.p.exper_id}-clean'
                else:
                    ckpt_dir_name = self.p.exper_id

            self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)
            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            fname_unet = '{}/{}.pt'.format(self.ckpt_dir, self.p.exper_id)
            print("where am i ?",type(stats['valid_loss'][epoch]),stats['valid_loss'][epoch])
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss)
        print('Saving checkpoint to: {}\n'.format(fname_unet))
        torch.save(self.model.state_dict(), fname_unet)
        print("where am i ?",type(stats['valid_loss'][epoch]),stats['valid_loss'][epoch])
        if stats['valid_psnr'][epoch] == max(stats['valid_psnr']):
            stats['best_epo_psnr'][0] = epoch
            stats['best_epo_psnr'][1] = stats['valid_psnr'][epoch]
            mybest = '{}/best-{}.pt'.format(self.ckpt_dir, "lucky")
            torch.save(self.model.state_dict(), mybest)

        # Save stats to JSON
        fname_dict = '{}/stats.json'.format(self.ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)


    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""

        print('Loading checkpoint from: {}'.format(ckpt_fname))
        if self.use_cuda:
            self.model.load_state_dict(torch.load(ckpt_fname))
        else:
            self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))


    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        """Tracks and saves starts after each epoch."""

        # Evaluate model on validation set
        print('\rTesting model on validation set... ', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr,valid_psnr_bsn = self.eval(valid_loader) 
        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)
        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr_bsn)

        # Decrease learning rate if plateau
        if self.p.scheduler == "plateau":
            self.scheduler.step(valid_loss) 
        if self.p.scheduler == "cos" and self.p.bigdataset == 0:
            self.scheduler.step()
        if self.p.scheduler == "step" and self.p.bigdataset == 0:
            self.scheduler.step()
            
        print("Learn_rate: %s /n" % self.optim.state_dict()['param_groups'][0]['lr'])
        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)
        stats['valid_psnr_bsn'].append(valid_psnr_bsn)
        self.save_model(epoch, stats, epoch == 0)

        # Plot stats
        if self.p.plot_stats:
            loss_str = f'{self.p.loss.upper()} loss'
            plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], loss_str)
            plot_per_epoch(self.ckpt_dir, 'Valid PSNR nbsn', stats['valid_psnr'], 'PSNR (dB)')
            plot_per_epoch(self.ckpt_dir, 'Valid PSNR bsn', stats['valid_psnr_bsn'], 'PSNR (dB)')
            plot_per_epoch(self.ckpt_dir, 'per iter train loss', stats['per_iter_train_loss'], "per iter train loss") 
        
        self.writer.add_scalars('valid/PSNR', {"nbsn":valid_psnr,"bsn":valid_psnr_bsn}, epoch)
        self.writer.add_scalars('Loss/Loss', {"valid_loss":valid_loss,"train_loss":train_loss}, epoch)

        
    def test(self, test_loader, show):

        self.model.train(False)

        source_imgs = []
        denoised_imgs = []
        clean_imgs = []

        # Create directory for denoised images
        denoised_dir = os.path.dirname(self.p.data)
        save_path = os.path.join(denoised_dir, 'res')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        
        test_start = datetime.now()
        for batch_idx, (source, target,gt) in enumerate(test_loader): 
            progress_bar(batch_idx, self.p.show_output, self.p.show_output, 123)
            with torch.no_grad():

                # Only do first <show> images
                if show == 0 or batch_idx >= show:
                    break

                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()

                if self.p.net == "csy":
                    if self.p.subtask == "double":
                        label=2
                        if self.p.crop_size==0:
                            nbsn,bsn,tfi,tfp = self.model(source)
                        else:
                            nbsn,bsn,tfi,tfp = self.model(source)
                        denoised_img = nbsn.clone()

                    
                        denoised_imgs.append(denoised_img.detach().cpu())    
                        source_imgs.append(target[:,label,:,:].unsqueeze_(dim=1).cpu())
                        clean_imgs.append(gt[:,label,:,:].unsqueeze_(dim=1).cpu()) 
                    
        test_elapsed = time_elapsed_since(test_start)[0]
        print('\n Test Total elapsed time: {}\n'.format(test_elapsed))
        
        # Squeeze tensors
        source_imgs = [t.squeeze(0) for t in source_imgs]
        denoised_imgs = [t.squeeze(0) for t in denoised_imgs]
        clean_imgs = [t.squeeze(0) for t in clean_imgs]

        # Create montage and save images
        print('Saving images and montages to: {}'.format(save_path))
        plot_start = datetime.now()

        if self.p.net == "csy":
            if self.p.subtask == "double":
                for i in tqdm(range(len(clean_imgs))):
                    img_name = test_loader.dataset.imgs_pair1[i]
                    create_montage(img_name, self.p.exper_id, save_path, source_imgs[i], denoised_imgs[i], clean_imgs[i], show)
                
        plot_elapsed = time_elapsed_since(plot_start)[0]
        print('Plot Total elapsed time: {}\n'.format(plot_elapsed))

    def eval(self, valid_loader):
        self.model.train(False)

        valid_start = datetime.now()
        loss_meter = AvgMeter()
        psnr_meter_nbsn = AvgMeter()
        psnr_meter_bsn = AvgMeter()

        for batch_idx, (source, target,gt) in enumerate(valid_loader):
            with torch.no_grad(): 
                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()
                    gt = gt.cuda()
                if self.p.net == "csy":
                    if self.p.subtask == "double":
                        if self.p.crop_size==0:
                            nbsn,bsn,tfi,tfp = self.model(source)
                            loss = self.loss(bsn,tfp)
                        else:
                            nbsn,bsn,tfi,tfp = self.model(source)
                            loss = 0.01*self.distill_loss(nbsn, bsn) + self.loss(bsn,tfi) #0.01-->1

                        source_denoised_nbsn = nbsn.clone()
                        source_denoised_bsn = bsn.clone()
                        print(source_denoised_nbsn.shape)
    
            loss_meter.update(loss.item())

            # Compute PSRN
            if self.p.net == "csy":
                if self.p.subtask == "double":
                    for i in range(source.shape[0]): 
                        gt = gt.cpu()
                        source_denoised_nbsn = source_denoised_nbsn.cpu()
                        source_denoised_bsn = source_denoised_bsn.cpu()
                        psnr_meter_nbsn.update(psnr(source_denoised_nbsn[i], gt[i][2].unsqueeze_(dim=0)).item())
                        psnr_meter_bsn.update(psnr(source_denoised_bsn[i], gt[i][2].unsqueeze_(dim=0)).item())

        valid_loss = loss_meter.avg
        valid_time = time_elapsed_since(valid_start)[0]
        psnr_avg_nbsn = psnr_meter_nbsn.avg
        psnr_avg_bsn = psnr_meter_bsn.avg

        return valid_loss, valid_time, psnr_avg_nbsn,psnr_avg_bsn

    def train(self, train_loader, valid_loader):

        self.model.train(True)
        
        self._print_params()
        num_batches = len(train_loader)
        assert num_batches % self.p.report_interval == 0, 'Report interval must divide total number of batches'
    
        # Dictionaries of tracked stats
        stats = {'exper_id': self.p.exper_id,
                 'lr':[],
                 'best_epo_psnr':[0,0],
                 'train_loss': [],
                 'valid_loss': [],
                 'per_iter_train_loss':[],
                 'valid_psnr': [],
                 'valid_psnr_bsn': [],
                }

        # Main training loop
        train_start = datetime.now()
        little_epoch=0
        for epoch in range(self.p.nb_epochs):
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))

            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            # Minibatch SGD
            for batch_idx, (source, target,gt) in enumerate(train_loader):
                batch_start = datetime.now()
                progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)
                
                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()
                    gt = gt.cuda()
                if self.p.net == "csy":
                    if self.p.subtask == "double":                   
                        nbsn,bsn,tfi,tfp = self.model(source)
#                         if epoch<75:
                        if epoch<0:
                            loss = self.loss(bsn,tfp)
                        else:
#                             loss = self.loss(bsn,tfp)
                            loss = 0.01*self.distill_loss(nbsn, bsn) + self.loss(bsn,tfi) #0.01-->1
                loss_meter.update(loss.item())
            
                # Zero gradients, perform a backward pass, and update the weights
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                    show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)            
                    stats['per_iter_train_loss'].append(loss_meter.avg)
                    stats['lr'].append(self.optim.state_dict()['param_groups'][0]['lr'])
                    if self.p.scheduler == "cos" and self.p.bigdataset != 0:
                        self.scheduler.step() 
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()

            # Epoch end, save and reset tracker
            if self.p.bigdataset == 0:
                self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
            train_loss_meter.reset()
            
        train_elapsed = time_elapsed_since(train_start)[0]
        print('Training done! Total elapsed time: {}\n'.format(train_elapsed))
