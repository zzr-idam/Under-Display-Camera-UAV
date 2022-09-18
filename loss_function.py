
import sys
import numpy as np
import torch
import torch.nn.functional as F


def alpha_total_variation(A):
    '''
    Links: https://remi.flamary.com/demos/proxtv.html
           https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/total_variation.html#total_variation
    '''
    delta_h = A[:, :, 1:, :] - A[:, :, :-1, :]
    delta_w = A[:, :, :, 1:] - A[:, :, :, :-1]

    # TV used here: L-1 norm, sum R,G,B independently
    # Other variation of TV loss can be found by google search
    tv = delta_h.abs().mean((2, 3)) + delta_w.abs().mean((2, 3))
    loss = torch.mean(tv.sum(1) / (A.shape[1] / 3))
    return loss


def exposure_control_loss(enhances, rsize=16, E=0.62):
    avg_intensity = F.avg_pool2d(enhances, rsize).mean(1)  # to gray: (R+G+B)/3
    exp_loss = (avg_intensity - E).abs().mean()
    return exp_loss


# Color constancy loss via gray-world assumption.   In use.
def color_constency_loss(enhances):
    plane_avg = enhances.mean((2, 3))
    col_loss = torch.mean((plane_avg[:, 0] - plane_avg[:, 1]) ** 2
                          + (plane_avg[:, 1] - plane_avg[:, 2]) ** 2
                          + (plane_avg[:, 2] - plane_avg[:, 0]) ** 2)
    return col_loss


# Averaged color component ratio preserving loss.  Not in use.
def color_constency_loss2(enhances, originals):
    enh_cols = enhances.mean((2, 3))
    ori_cols = originals.mean((2, 3))
    rg_ratio = (enh_cols[:, 0] / enh_cols[:, 1] - ori_cols[:, 0] / ori_cols[:, 1]).abs()
    gb_ratio = (enh_cols[:, 1] / enh_cols[:, 2] - ori_cols[:, 1] / ori_cols[:, 2]).abs()
    br_ratio = (enh_cols[:, 2] / enh_cols[:, 0] - ori_cols[:, 2] / ori_cols[:, 0]).abs()
    col_loss = (rg_ratio + gb_ratio + br_ratio).mean()
    return col_loss


# pixel-wise color component ratio preserving loss. Not in use.
def anti_color_shift_loss(enhances, originals):
    def solver(c1, c2, d1, d2):
        pos = (c1 > 0) & (c2 > 0) & (d1 > 0) & (d2 > 0)
        return torch.mean((c1[pos] / c2[pos] - d1[pos] / d2[pos]) ** 2)

    enh_avg = F.avg_pool2d(enhances, 4)
    ori_avg = F.avg_pool2d(originals, 4)

    rg_loss = solver(enh_avg[:, 0, ...], enh_avg[:, 1, ...],
                     ori_avg[:, 0, ...], ori_avg[:, 1, ...])
    gb_loss = solver(enh_avg[:, 1, ...], enh_avg[:, 2, ...],
                     ori_avg[:, 1, ...], ori_avg[:, 2, ...])
    br_loss = solver(enh_avg[:, 2, ...], enh_avg[:, 0, ...],
                     ori_avg[:, 2, ...], ori_avg[:, 0, ...])

    anti_shift_loss = rg_loss + gb_loss + br_loss
    if torch.any(torch.isnan(anti_shift_loss)).item():
        sys.exit('Color Constancy loss is nan')
    return anti_shift_loss



def spatial_consistency_loss(enhances, originals, to_gray, neigh_diff, rsize=4):
    # convert to gray
    enh_gray = F.conv2d(enhances, to_gray)
    ori_gray = F.conv2d(originals, to_gray)

    # average intensity of local regision
    enh_avg = F.avg_pool2d(enh_gray, rsize)
    ori_avg = F.avg_pool2d(ori_gray, rsize)

    # calculate spatial consistency loss via convolution
    enh_pad = F.pad(enh_avg, (1, 1, 1, 1), mode='replicate')
    ori_pad = F.pad(ori_avg, (1, 1, 1, 1), mode='replicate')
    enh_diff = F.conv2d(enh_pad, neigh_diff)
    ori_diff = F.conv2d(ori_pad, neigh_diff)

    spa_loss = torch.pow((enh_diff - ori_diff), 2).sum(1).mean()
    return spa_loss

def gamma_correction(img, gamma):
    return np.power(img, gamma)


import torch
import numpy as np
import torch.nn as nn
from torch.nn import L1Loss




# https://github.com/Prevalenter/semi-dehazing
# calculationg dark channel of image : https://github.com/joyeecheung/dark-channel-prior-dehazing/tree/master/src

def DCLoss(img, opt):
    """
    calculating dark channel of image, the image shape is of N*C*W*H
    """
    maxpool = nn.MaxPool3d((3, opt.patch_size, opt.patch_size), stride=1, padding=(0, opt.patch_size//2, opt.patch_size//2))
    dc = maxpool(1-img[:, None, :, :, :])
    
    target = torch.FloatTensor(dc.shape).zero_().cuda(opt.gpu_ids[0])
     
    loss = L1Loss(reduction='sum')(dc, target)
    return -loss

# total loss DCLoss is key moduel.
