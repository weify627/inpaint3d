"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    L2 loss implementation
"""


import torch
import torch.nn as nn


def get_mask(validmin, validmax, gt, ellipsis):
    mask = gt > validmin
    if validmax > 0:
        mask &= gt < validmax
    if ellipsis:
        # pause() #dtype, cpu/gpu, value
        h, w = gt.shape[-2:]
        x = torch.arange(w) - (w - 1) / 2
        y = torch.arange(h) - (h - 1) / 2
        grid_x, grid_y = torch.meshgrid(x, y)
        mask_e = ((grid_x/ (w/2))**2 + (grid_y/(h/2))**2) <= 1
        mask &= mask_e.T[None, None].cuda()
    return mask


class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()

        self.args = args
        self.t_valid = 0.0001
        self.t_validmax = args.t_validmax
        self.ellipsis = args.mask_ellipsis
        # self.get_mask = get_mask

    def forward(self, pred, gt):
        gt = torch.clamp(gt, min=0, max=self.args.max_depth)
        pred = torch.clamp(pred, min=0, max=self.args.max_depth)

        # mask = (gt > self.t_valid).type_as(pred).detach()
        # mask = gt > self.t_valid
        # if self.t_validmax > 0:
            # mask &= gt < self.t_validmax
        mask = get_mask(self.t_valid, self.t_validmax, gt, self.ellipsis)
        mask = mask.type_as(pred).detach()

        d = torch.pow(pred - gt, 2) * mask

        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask, dim=[1, 2, 3])

        loss = d / (num_valid + 1e-8)

        loss = loss.sum()

        return loss
