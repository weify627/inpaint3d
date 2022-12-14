"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    NLSPNMetric implementation
"""


import torch
from . import BaseMetric
from pdb import set_trace as pause


def get_mask(validmin, validmax, gt, ellipsis):
    mask = gt > validmin
    if validmax > 0:
        mask &= gt < validmax
    if ellipsis:
        h, w = gt.shape[-2:]
        x = torch.arange(w) - (w - 1) / 2
        y = torch.arange(h) - (h - 1) / 2
        grid_x, grid_y = torch.meshgrid(x, y)
        mask_e = ((grid_x/ (w/2))**2 + (grid_y/(h/2))**2) <= 1
        mask &= mask_e.T[None, None].cuda()
    return mask


class NLSPNMetric(BaseMetric):
    def __init__(self, args):
        super(NLSPNMetric, self).__init__(args) #, get_mask)

        self.args = args
        self.t_valid = 0.0001
        self.t_validmax = args.t_validmax
        self.ellipsis = args.mask_ellipsis
        # self.get_mask = get_mask

        self.metric_name = [
            'RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3'
        ]

    def evaluate(self, sample, output, mode):
        with torch.no_grad():
            pred = output['pred'].detach()
            gt = sample['gt'].detach()

            pred_inv = 1.0 / (pred + 1e-8)
            gt_inv = 1.0 / (gt + 1e-8)

            # For numerical stability
            # mask = gt > self.t_valid
            # if self.t_validmax > 0:
                # mask &= gt < self.t_validmax
            mask = get_mask(self.t_valid, self.t_validmax, gt, self.ellipsis)
            num_valid = mask.sum()

            pred = pred[mask]
            gt = gt[mask]

            pred_inv = pred_inv[mask]
            gt_inv = gt_inv[mask]

            # gt_inv[~mask] = 0.0
            # maskpred = get_mask(self.t_valid, self.t_validmax, pred, self.ellipsis)
            # pred_inv[~maskpred] = 0.0
            #fixme
            pred_inv[pred <= self.t_valid] = 0.0
            gt_inv[gt <= self.t_valid] = 0.0
            # if self.t_validmax > 0:
                # pred_inv[pred >= self.t_validmax] = 0.0
                # gt_inv[gt >= self.t_validmax] = 0.0

            # RMSE / MAE
            diff = pred - gt
            diff_abs = torch.abs(diff)
            diff_sqr = torch.pow(diff, 2)

            rmse = diff_sqr.sum() / (num_valid + 1e-8)
            rmse = torch.sqrt(rmse)

            mae = diff_abs.sum() / (num_valid + 1e-8)

            # iRMSE / iMAE
            diff_inv = pred_inv - gt_inv
            diff_inv_abs = torch.abs(diff_inv)
            diff_inv_sqr = torch.pow(diff_inv, 2)

            irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
            irmse = torch.sqrt(irmse)

            imae = diff_inv_abs.sum() / (num_valid + 1e-8)

            # Rel
            rel = diff_abs / (gt + 1e-8)
            rel = rel.sum() / (num_valid + 1e-8)

            # delta
            r1 = gt / (pred + 1e-8)
            r2 = pred / (gt + 1e-8)
            ratio = torch.max(r1, r2)

            del_1 = (ratio < 1.25).type_as(ratio)
            del_2 = (ratio < 1.25**2).type_as(ratio)
            del_3 = (ratio < 1.25**3).type_as(ratio)

            del_1 = del_1.sum() / (num_valid + 1e-8)
            del_2 = del_2.sum() / (num_valid + 1e-8)
            del_3 = del_3.sum() / (num_valid + 1e-8)

            result = [rmse, mae, irmse, imae, rel, del_1, del_2, del_3]
            result = torch.stack(result)
            result = torch.unsqueeze(result, dim=0).detach()

        return result
