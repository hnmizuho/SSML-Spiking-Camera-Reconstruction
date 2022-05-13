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
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of self-supervised mutual learning for spiking camera)')

    # Data parameters
    parser.add_argument('-t', '--train-dir', help='training set path', default='./../data/train')
    parser.add_argument('-v', '--valid-dir', help='test set path', default='./../data/valid')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../ckpts')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save',type=int)
    parser.add_argument('--report-interval', help='batch report interval', default=500, type=int)
    parser.add_argument('-ts', '--train-size', help='size of train dataset', type=int)
    parser.add_argument('-vs', '--valid-size', help='size of valid dataset', type=int)

    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.999, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=4, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2', 'hdr'], default='l1', type=str)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')

    parser.add_argument('-n', '--exper-id', help='experiment id',default='omg', type=str)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=128, type=int)
    parser.add_argument('--clean-targets', help='use clean targets for training', action='store_true')
    
    parser.add_argument('--net', help='which net,n2n,s2s,n2v,rotate,vidar,vidar2,vidar3,vidar4,vidar20,vidar_udvd,vidar20g,vidar30g,zj,csy', default='csy')
    parser.add_argument('--subtask', help='used for csy net')
    
    parser.add_argument('--use_resume', help='resume model checkpoint')
    parser.add_argument('--resume', help='resume model checkpoint')
    
    parser.add_argument('--bigdataset', help='big or small, this is a question',type=int)
    parser.add_argument('--scheduler', help='plateau,cos',default='plateau',type=str)
    
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
    seed_torch(1314)
    # Parse training parameters
    params = parse_args()
    
    # Train/valid datasets
    train_loader = load_dataset(params.train_dir, params.train_size, params, params.crop_size, shuffled=True)
    valid_loader = load_dataset(params.valid_dir, params.valid_size, params, params.crop_size, shuffled=False) 

    # Initialize model and train
    trainer = Trainer(params, trainable=True)
    
    if params.use_resume==1:
        print("using resume!!!!!!!!!")
        trainer.load_model(params.resume)
    
    trainer.train(train_loader, valid_loader)