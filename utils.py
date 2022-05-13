#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
import torch.nn as nn

import os
import numpy as np
import math
from math import log10
from datetime import datetime
from PIL import Image

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import cv2

def clear_line():
    """Clears line from any characters."""

    print('\r{}'.format(' ' * 80), end='\r')


def progress_bar(batch_idx, num_batches, report_interval, train_loss):
    """Neat progress bar to track training."""

    dec = int(np.ceil(np.log10(num_batches)))
    bar_size = 21 + dec
    progress = (batch_idx % report_interval) / report_interval
    fill = int(progress * bar_size) + 1
    print('\rBatch {:>{dec}d} [{}{}] Train loss: {:>1.5f}'.format(batch_idx + 1, '=' * fill + '>', ' ' * (bar_size - fill), train_loss, dec=str(dec)), end='')


def time_elapsed_since(start):
    """Computes elapsed time since start."""

    timedelta = datetime.now() - start
    string = str(timedelta)[:-7]
    ms = int(timedelta.total_seconds() * 1000)

    return string, ms


def show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr):
    """Formats validation error stats."""

    clear_line()
    print('Train time: {} | Valid time: {} | Valid loss: {:>1.5f} | Avg PSNR: {:.2f} dB'.format(epoch_time, valid_time, valid_loss, valid_psnr))


def show_on_report(batch_idx, num_batches, loss, elapsed):
    """Formats training stats."""

    clear_line()
    dec = int(np.ceil(np.log10(num_batches)))
    print('Batch {:>{dec}d} / {:d} | Avg loss: {:>1.5f} | Avg train time / batch: {:d} ms'.format(batch_idx + 1, num_batches, loss, int(elapsed), dec=dec))


def plot_per_epoch(ckpt_dir, title, measurements, y_label):
    """Plots stats (train/valid loss, avg PSNR, etc.)."""

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, len(measurements) + 1), measurements)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()

    fname = '{}.png'.format(title.replace(' ', '-').lower())
    plot_fname = os.path.join(ckpt_dir, fname)
    plt.savefig(plot_fname, dpi=200)
    plt.close()


def reinhard_tonemap(tensor):
    """Reinhard et al. (2002) tone mapping."""

    tensor[tensor < 0] = 0
    return torch.pow(tensor / (1 + tensor), 1 / 2.2)


def psnr(input, target):
    """Computes peak signal-to-noise ratio."""
    
    return 10 * torch.log10(1 / F.mse_loss(input, target))


def create_montage(img_name, exper_id, save_path, source_t, denoised_t, clean_t, show):
    """Creates montage for easy comparison."""
    # Bring tensors to CPU
    c = source_t.shape[0]
    denoised_t = denoised_t.cpu()
    denoised = tvF.to_pil_image(torch.clamp(denoised_t, 0, 1)) 

    # Open pop up window, if requested
    if show > 0:
        plt.show()

    # Save to files
    fname = os.path.splitext(img_name)[0]
#     source.save(os.path.join(save_path, f'{fname}-{exper_id}-noisy.png'))
    denoised.save(os.path.join(save_path, f'{fname}-{exper_id}-denoised.png'))
#     fig.savefig(os.path.join(save_path, f'{fname}-{exper_id}-montage.png'), bbox_inches='tight')

class AvgMeter(object):
    """Computes and stores the average and current value.
    Useful for tracking averages such as elapsed times, minibatch losses, etc.
    """

    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0.
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def smooth_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()) * weight
        weight /= 2.0
    return loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss
    
def load_vidar_dat(filename, left_up=(0, 0), window=None, frame_cnt = None, **kwargs):
    if isinstance(filename, str):
        array = np.fromfile(filename, dtype=np.uint8)
    elif isinstance(filename, (list, tuple)):
        l = []
        for name in filename:
            a = np.fromfile(name, dtype=np.uint8)
            l.append(a)
        array = np.concatenate(l)
    else:
        raise NotImplementedError
    
    height = 250
    width = 400

    if window == None:
        window = (height - left_up[0], width - left_up[0])

    len_per_frame = height * width // 8
    framecnt = frame_cnt if frame_cnt != None else len(array) // len_per_frame

    spikes = []

    for i in range(framecnt):
        compr_frame = array[i * len_per_frame: (i + 1) * len_per_frame]
        blist = []
        for b in range(8):
            blist.append(np.right_shift(np.bitwise_and(compr_frame, np.left_shift(1, b)), b))
        
        frame_ = np.stack(blist).transpose()
        frame_ = np.flipud(frame_.reshape((height, width), order='C'))

        if window is not None:
            spk = frame_[left_up[0]:left_up[0] + window[0], left_up[1]:left_up[1] + window[1]]
        else:
            spk = frame_

        spk = torch.from_numpy(spk.copy().astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0)

        spikes.append(spk)

    return torch.cat(spikes) 

import cv2
def TFI(spike, frame_cnt=400,name = "S"):
#     spike = spike.squeeze(1).numpy()
    #初版，速度较慢
    mask = np.zeros_like(spike[0],dtype=int)
    for i in range(1,frame_cnt):
        curr = np.nonzero(spike[i])
        spike[mask[curr],curr[0],curr[1]] = 255.0/(i-mask)[curr] 
        mask[curr] = i
    for i in range(frame_cnt):
        curr = np.where(spike[i]==0)
        spike[i,curr[0],curr[1]] = spike[i-1,curr[0],curr[1]]
        curr = np.where(spike[i]==1) 
        spike[i,curr[0],curr[1]] = spike[i-1,curr[0],curr[1]]
    
    spike = spike.astype(np.uint8) 
    return spike

def middleTFI(spike, middle, window=50):
    #左右找1
    spike = spike.squeeze(1).numpy() 
    C, H, W = spike.shape
    lindex, rindex = np.zeros([H, W]), np.zeros([H, W])
    l, r = middle+1, middle+1
    for r in range(middle+1, middle + window+1): #往左包括自己50个,往右不包括自己也是50个
        l = l - 1
        if l>=0:
            newpos = spike[l, :, :]*(1 - np.sign(lindex)) 
            distance = l*newpos
            lindex += distance
        if r<C:
            newpos = spike[r, :, :]*(1 - np.sign(rindex))
            distance = r*newpos
            rindex += distance

    rindex[rindex==0] = window+middle
    lindex[lindex==0] = middle-window
    interval = rindex - lindex
    tfi = 1.0 / interval
    
    return tfi

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)    
        
class Gradient_Net(nn.Module):
    def __init__(self):
        super(Gradient_Net, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()

        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x)
        grad_y = F.conv2d(x, self.weight_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient
def fo_gradient(x):
    gradient_model = Gradient_Net().cuda()
    g = gradient_model(x)
    return g

def calculate_x_gradient(images):
    x_gradient_filter = torch.Tensor(
        [
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
        ]
    ).cuda()
    x_gradient_filter = x_gradient_filter.view(3, 1, 3, 3)
    result = torch.functional.F.conv2d(
        images, x_gradient_filter, groups=3, padding=(1, 1)
    )
    return result


def calculate_y_gradient(images):
    y_gradient_filter = torch.Tensor(
        [
            [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
            [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
            [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
        ]
    ).cuda()
    y_gradient_filter = y_gradient_filter.view(3, 1, 3, 3)
    result = torch.functional.F.conv2d(
        images, y_gradient_filter, groups=3, padding=(1, 1)
    )
    return result

def loss_igdl( correct_images, generated_images): # taken from https://github.com/Arquestro/ugan-pytorch/blob/master/ops/loss_modules.py
    correct_images_gradient_x = calculate_x_gradient(correct_images)
    generated_images_gradient_x = calculate_x_gradient(generated_images)
    correct_images_gradient_y = calculate_y_gradient(correct_images)
    generated_images_gradient_y = calculate_y_gradient(generated_images)
    pairwise_p_distance = torch.nn.PairwiseDistance(p=1)
    distances_x_gradient = pairwise_p_distance(
        correct_images_gradient_x, generated_images_gradient_x
    )
    distances_y_gradient = pairwise_p_distance(
        correct_images_gradient_y, generated_images_gradient_y
    )
    loss_x_gradient = torch.mean(distances_x_gradient)
    loss_y_gradient = torch.mean(distances_y_gradient)
    loss = 0.5 * (loss_x_gradient + loss_y_gradient)
    return loss

def normalize(x):
    return (x-x.min()) / (x.max() - x.min())

if __name__ == '__main__':
#     a=torch.ones(4,1,40,40).cuda() 
#     print(fo_gradient(a).shape) #4 1 38 38
    show_on_epoch_end(0, 0, 0, 0)
    show_on_epoch_end(0, 0, 0, 0)