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
import sys
import warnings
import numpy as np
import json
import h5py
from . import BaseDataset

from PIL import Image
import skimage
import cv2
import imageio
from scipy.interpolate import NearestNDInterpolator
import scipy.ndimage as ndi
import scipy
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from pdb import set_trace as pause
from data.util import get_remapper, read_label_mapping, get_ids_uni

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

def load_label(path_rgb):
    label_path = path_rgb.replace('scannetv2_images', 'label_filto').replace("/color", "_2d-label-filt.zip/label-filt")
    label_path = label_path.replace('jpg', 'png')
    label = imageio.imread(label_path)
    labelraw = skimage.transform.resize(label, [240, 320], order=0, preserve_range=True, anti_aliasing=False).astype(np.int16)
    return labelraw

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


def sort_glob(path):
    from glob import glob
    files = glob(path+"/*")
    files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0])) 
    return files



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
        if args.split_json[-4:] == "json":
            with open(self.args.split_json) as json_file:
                json_data = json.load(json_file)
                self.sample_list = json_data[mode]
        else:
            rgb_list = sort_glob(f"{args.split_json}/color")
            depth_list = sort_glob(f"{args.split_json}/depthhole")
            gt_list = depth_list
            pause() #print(two lists
            assert len(rgb_list) == len(depth_list)
            assert len(rgb_list) == len(gt_list)
            K = f"/data/fwei/scannet/ScanNet/SensReader/python/scannetv2_images/scene0{args.scene}/intrinsic/intrinsic_color.txt"
            self.sample_list =[{"rgb":rgb, "depth": depth, "gt": gt, "K": K} \
                    for (rgb, depth, gt) in zip(rgb_list, depth_list, gt_list)]


        if self.args.label_mask:
            path_to_module = "/n/fs/rgbd/users/fwei/data/scannet/data/tools"
            sys.path.insert(0, path_to_module)
            # sys.argv.append('--ClutterSize')
            # sys.argv.append("small")
            import ids
            # print(sys.argv)
            # sys.argv.pop(-1)
            # sys.argv.pop(-1)
            # print(sys.argv)

            self.pureclabelexclude = ids.pureclabelexclude
            label_map_file = "/data/fwei/scannet/data/scannetv2-labels.combined.tsv"
            label_map = read_label_mapping(label_map_file, label_from='raw_category', label_to='id')
            ids_uni = get_ids_uni(label_map)
            self.ids_uni = ids_uni
            ignore_label = 1000
            self.remapper = get_remapper(ids_uni, ignore_label=ignore_label)
            # len(self.remapper) = 1358, len(self.remapper3d) = 1001
            for i, v in enumerate(self.remapper):
                if v == ignore_label: continue
                if i in ids.clutter_rawid:
                    self.remapper[i] = 0  # commented for evaluation
                else:
                    self.remapper[i] = 1


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        rgb, dep, gt, K, label = self._load_data(idx)

        if self.augment and self.mode == 'train':
            _scale = np.random.uniform(1.0, 1.5)
            scale = np.int(self.height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)
                gt = TF.hflip(gt)
                if self.args.label_mask:
                    label = TF.hflip(label)
                #fixme
                # K[2] = width - K[2]

            rgb = TF.rotate(rgb, angle=degree, resample=Image.NEAREST)
            dep = TF.rotate(dep, angle=degree, resample=Image.NEAREST)
            gt = TF.rotate(gt, angle=degree, resample=Image.NEAREST)
            if self.args.label_mask:
                label = TF.rotate(label, angle=degree, resample=Image.NEAREST)

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
            gt = t_dep(gt)
            if self.args.label_mask:
                label = t_dep(label)

            dep = dep / _scale
            gt = gt / _scale

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
            gt = t_dep(gt)

            if self.args.label_mask:
                label = t_dep(label)

            # K = self.K.clone()

        # fixme sample?
        # data = dep[0] * 1
        # maskidx = np.where(data>0)
        # interp = NearestNDInterpolator(np.transpose(maskidx), data[maskidx])
        # filled_data = interp(*np.indices(data.shape))
        # mean_kernel = np.full((7, 7), 1/49)
        # mean_data = ndi.correlate(filled_data, mean_kernel)
        # mask = (data>0).float()
        # dep = torch.Tensor(mean_data) * (1 - mask) + data * mask
        # dep = dep[None]
        if self.args.label_mask:
            # label_mask = self.remapper[label]
            # if (label_mask==0).sum() == 0:
                # print(self.sample_list[idx]['rgb'])
            # dep_sp = self.get_masked_depth(dep, label)
            dep_sp = dep * label
        else:
            dep_sp = self.get_sparse_depth(dep, self.args.num_sample)
        output = {'rgb': rgb, 'dep': dep_sp, 'gt': gt, 'K': torch.Tensor(K), \
                'dep_ori': dep}
                # 'dep_init': dep_init[None]}

        return output


    def _load_data(self, idx):
        path_rgb = os.path.join(self.args.dir_data,
                                self.sample_list[idx]['rgb'])
        path_depth = os.path.join(self.args.dir_data,
                                  self.sample_list[idx]['depth'])
        path_gt = os.path.join(self.args.dir_data,
                               self.sample_list[idx]['gt'])
        path_calib = os.path.join(self.args.dir_data,
                                  self.sample_list[idx]['K'])
        if self.args.label_mask:
            # label_mask = self.remapper[label]
            # if (label_mask==0).sum() == 0:
                # print(self.sample_list[idx]['rgb'])
            label = load_label(path_rgb)
            label = self.get_depth_mask(label)
            label = Image.fromarray(label.astype(np.int32), mode='I')
        else:
            label = None
        #fixme .replace('color','depth')

        depth = read_depth(path_depth)
        gt = read_depth(path_gt)

        # depth = cv2.copyMakeBorder(depth, 6, 6, 8, 8, cv2.BORDER_REPLICATE) 
        rgb = Image.open(path_rgb)
        depth = Image.fromarray(depth.astype('float32'), mode='F')
        gt = Image.fromarray(gt.astype('float32'), mode='F')
        # gt = depth * 1.0

        # if self.mode in ['train', 'val']:
        calib = read_calib_file(path_calib)
        K_cam = np.reshape(calib, (4, 4))
        #fixme
        K = [K_cam[0, 0]/2, K_cam[1, 1]/2, K_cam[0, 2]/2 - 8.0, K_cam[1, 2]/2 - 6.0] # account for crop
        # print(np.array(depth).max())

        w1, h1 = rgb.size
        w2, h2 = depth.size
        w3, h3 = gt.size

        assert w1 == w2 and h1 == h2
        assert w1 == w3 and h1 == h3

        return rgb, depth, gt, K, label

    def n_sample(self, n_total, seed, size=None, replace=False):
        if replace == None:
            replace = n_total < size
        if self.mode == 'train':
            ret = np.random.choice(n_total, size, replace=replace)
        else:
            ret = np.random.RandomState(seed).choice(n_total, size, replace=replace)
        return ret

    def get_depth_mask(self, label):
        label_mask = self.remapper[label]
        label_mask[label_mask==1000]=1
        if (label_mask==0).sum() <40000:
            is_found = False
            for i in range(20):
                idx_s = self.n_sample(len(self.sample_list),label.sum()+i)
                path_rgb = os.path.join(self.args.dir_data,
                                self.sample_list[idx_s]['rgb'])
                label_ = load_label(path_rgb)
                label_mask_ = self.remapper[label_]
                label_mask_[label_mask_==1000]=1
                if (label_mask_==0).sum() != 0:
                    is_found = True
                    break
            if is_found:
                n_iter = self.n_sample(20, np.array(label).sum())+3
                label_mask_ = 1-scipy.ndimage.binary_dilation(1-label_mask_, iterations=n_iter)
                n_iter = self.n_sample(20, np.array(label).sum())+3
                label_mask = 1-scipy.ndimage.binary_dilation(1-label_mask, iterations=n_iter)
                depth_mask = (1-label_mask).astype(bool) | label_mask_.astype(bool)
                if depth_mask.sum()<(label.shape[0]*label.shape[1]/4):
                    is_found = False
            if not is_found:
                label_mask_ = label_mask * 1
                ud = self.n_sample(label.shape[0], label.sum(), 2)
                lr = self.n_sample(label.shape[1], label.sum(), 2)
                label_mask_[ud.min():ud.max(),lr.min():lr.max()]=0
                depth_mask = (1-label_mask).astype(bool) | label_mask_.astype(bool)
        else:
            n_iter = self.n_sample(30, np.array(label).sum())+10
            depth_mask = scipy.ndimage.binary_dilation(1-label_mask, iterations=n_iter)
        return depth_mask
        dep_sp = dep * depth_mask
        return dep_sp

    def get_sparse_depth(self, dep, num_sample):
        # num_sample = dep.shape[1] * dep.shape[2] // (dep==0).sum()
        if num_sample == 0:
            return dep
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
