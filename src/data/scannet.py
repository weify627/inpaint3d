"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    SCANNET Depth V2 Dataset Helper
"""


import os
import warnings
import numpy as np
import json
import h5py
from . import BaseDataset

from PIL import Image
# from scipy.interpolate import NearestNDInterpolator
import scipy.ndimage as ndi
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from pdb import set_trace as pause

warnings.filterwarnings("ignore", category=UserWarning)

"""
SCANNETDepthV2 json file has a following format:

{
    "train": [
        {
            "filename": "train/bedroom_0078/00066.h5"
        }, ...
    ],
    "val": [
        {
            "filename": "train/study_0008/00351.h5"
        }, ...
    ],
    "test": [
        {
            "filename": "val/official/00001.h5"
        }, ...
    ]
}

Reference : https://github.com/XinJCheng/CSPN/blob/master/nyu_dataset_loader.py
"""

def read_depth(file_name):
    # loads depth map D from 16 bits png file as a numpy array,
    # refer to readme file in KITTI dataset
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.array(Image.open(file_name))

    # Consider empty depth
    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
        "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    image_depth = image_depth.astype(np.float32) / 1000.0
    return image_depth


# Reference : https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
def read_calib_file(filepath):
    """Read in a calibration file and parse into a numpy array."""

    with open(filepath, 'r') as f:
        lines = f.readlines()
        lines = [list(map(float, line.split())) for line in lines]
    data = np.array(lines)
    return data


class SCANNET(BaseDataset):
    def __init__(self, args, mode):
        super(SCANNET, self).__init__(args, mode)

        self.args = args
        self.mode = mode

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        # For SCANNETDepthV2, crop size is fixed
        height, width = (240, 320)
        crop_size = (228, 304)

        self.height = height
        self.width = width
        self.crop_size = crop_size

        # Camera intrinsics [fx, fy, cx, cy]
        # fixme
        # self.K = torch.Tensor([
            # 5.1885790117450188e+02 / 2.0,
            # 5.1946961112127485e+02 / 2.0,
            # 3.2558244941119034e+02 / 2.0 - 8.0,
            # 2.5373616633400465e+02 / 2.0 - 6.0
        # ])

        self.augment = self.args.augment

        with open(self.args.split_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        rgb, dep, K = self._load_data(idx)

        if self.augment and self.mode == 'train':
            _scale = np.random.uniform(1.0, 1.5)
            scale = np.int(self.height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)
                #fixme
                # K[2] = width - K[2]

            rgb = TF.rotate(rgb, angle=degree, resample=Image.NEAREST)
            dep = TF.rotate(dep, angle=degree, resample=Image.NEAREST)

            t_rgb = T.Compose([
                T.Resize(scale),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                #fixme
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            t_dep = T.Compose([
                T.Resize(scale),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

            dep = dep / _scale

            # K = self.K.clone()
            K[0] = K[0] * _scale
            K[1] = K[1] * _scale
            #fixme 2 Ks
        else:
            t_rgb = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            t_dep = T.Compose([
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

            # K = self.K.clone()

        # fixme sample?
        dep_sp = self.get_sparse_depth(dep, self.args.num_sample)
        # data = dep_sp[0] * 1
        # maskidx = np.where(data>0)
        # interp = NearestNDInterpolator(np.transpose(maskidx), data[maskidx])
        # filled_data = interp(*np.indices(data.shape))
        # mean_kernel = np.full((7, 7), 1/49)
        # mean_data = ndi.correlate(filled_data, mean_kernel)
        # mask = (data>0).float()
        # dep_init = torch.Tensor(mean_data) * (1 - mask) + data * mask

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': torch.Tensor(K), \
                'dep_ori': dep}
                # 'dep_init': dep_init[None]}

        return output

    def _load_data(self, idx):
        path_rgb = os.path.join(self.args.dir_data,
                                self.sample_list[idx]['rgb'])
        path_depth = os.path.join(self.args.dir_data,
                                  self.sample_list[idx]['depth'])
        # path_gt = os.path.join(self.args.dir_data,
                               # self.sample_list[idx]['gt'])
        path_calib = os.path.join(self.args.dir_data,
                                  self.sample_list[idx]['K'])
        #fixme .replace('color','depth')

        depth = read_depth(path_depth)
        # gt = read_depth(path_gt)

        rgb = Image.open(path_rgb)
        depth = Image.fromarray(depth.astype('float32'), mode='F')
        # gt = Image.fromarray(gt.astype('float32'), mode='F')
        # gt = depth * 1.0

        # if self.mode in ['train', 'val']:
        calib = read_calib_file(path_calib)
        K_cam = np.reshape(calib, (4, 4))
        K = [K_cam[0, 0], K_cam[1, 1], K_cam[0, 2] - 8.0, K_cam[1, 2] - 6.0] # account for crop
        # print(np.array(depth).max())

        w1, h1 = rgb.size
        w2, h2 = depth.size
        # w3, h3 = gt.size

        assert w1 == w2 and h1 == h2

        return rgb, depth, K

    def get_sparse_depth(self, dep, num_sample):
        # num_sample = dep.shape[1] * dep.shape[2] // (dep==0).sum()
        # return dep
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)

        return dep_sp
