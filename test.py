#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from datasets import load_dataset
from main import Trainer

from argparse import ArgumentParser
import random
import numpy as np
import os

def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of self-supervised mutual learning for spiking camera)')

    # Data parameters
    parser.add_argument('-d', '--data', help='dataset root path', default='../data')
    parser.add_argument('--load-ckpt', help='load model checkpoint')
    parser.add_argument('--show-output', help='pop up window to display outputs', default=0, type=int)
    parser.add_argument('--cuda', help='use cuda', action='store_true')

    # Corruption parameters

    parser.add_argument('-n', '--exper-id', help='experiment id',default='omg', type=str)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='image crop size', default=256, type=int)
    
    parser.add_argument('--net', help='which net,n2n,s2s,n2v,rotate,vidar,vidar2,vidar3,vidar4,vidar20,vidar_udvd,vidar20g,vidar30g,zj,csy', default='csy')
    parser.add_argument('--pingce', help='1 use pingce,else test', type=int)
    parser.add_argument('--subtask', help='used for csy net')
    
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2', 'hdr'], default='l2', type=str)
    parser.add_argument('--scheduler', help='plateau,cos',default='plateau',type=str)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.999, 1e-8], type=list)
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=2, type=int)
    parser.add_argument('--bigdataset', help='big or small, this is a question',type=int,default=0)
    parser.add_argument('--valid-dir', help='test set path', default='./../data/valid') 
    parser.add_argument('--valid-size', help='size of valid dataset', type=int)
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../ckpts')
    
    return parser.parse_args()

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
if __name__ == '__main__':
    seed_torch(0) 
    # Parse test parameters
    params = parse_args()

    # Initialize model and test
    trainer = Trainer(params, trainable=True)
    params.redux = False
    params.clean_targets = True

    test_loader = load_dataset(params.data, 0, params, params.crop_size, shuffled=False, single=True) 
    trainer.load_model(params.load_ckpt)
    
    if params.pingce == 1:
        valid_loader = load_dataset(params.valid_dir, params.valid_size, params, params.crop_size, shuffled=False)
        print('\rTesting model on validation set... ', end='')
        valid_loss, valid_time, valid_psnr,valid_psnr_bsn = trainer.eval(valid_loader)
        print('Valid time: {} | Valid loss: {:>1.5f} | Avg PSNR: {:.2f} dB'.format(valid_time, valid_loss, valid_psnr))
        print('Valid time: {} | Valid loss: {:>1.5f} | Avg PSNR: {:.2f} dB'.format(valid_time, valid_loss, valid_psnr_bsn))
    else:
        trainer.test(test_loader, show=params.show_output)
        

