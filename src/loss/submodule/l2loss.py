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


class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()

        self.args = args
        self.t_valid = 0.0001
        self.t_validmax = args.t_validmax
        print("L2", self.t_validmax)

    def forward(self, pred, gt):
        gt = torch.clamp(gt, min=0, max=self.args.max_depth)
        pred = torch.clamp(pred, min=0, max=self.args.max_depth)

        # mask = (gt > self.t_valid).type_as(pred).detach()
        mask = gt > self.t_valid
        if self.t_validmax > 0:
            mask &= gt < self.t_validmax
        mask = mask.type_as(pred).detach()

        d = torch.pow(pred - gt, 2) * mask

        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask, dim=[1, 2, 3])

        loss = d / (num_valid + 1e-8)

        loss = loss.sum()

        return loss
