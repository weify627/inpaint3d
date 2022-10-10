
"""
    This script generates a json file for the NYUDepthV2 HDF5 dataset.
"""


import os
from glob import glob
import random

import argparse
import json
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description="SCANNET json generator")

parser.add_argument('--path_root', type=str, required=True,
                    help="Path to SCANNET dataset")

parser.add_argument('--path_out', type=str, required=False,
                    default='../data_json', help="Output path")
parser.add_argument('--name_out', type=str, required=False,
                    default='scannet.json', help='Output file name')
parser.add_argument('--val_ratio', type=float, required=False,
                    default=0.05, help='Validation data ratio')
parser.add_argument('--txt_train', type=str, required=False,
                    default='/data/fwei/scannet/ScanNet/Tasks/Benchmark/scannetv2_train.txt',
                    help='Train data csv file')
parser.add_argument('--txt_test', type=str, required=False,
                    default='/data/fwei/scannet/ScanNet/Tasks/Benchmark/scannetv2_valmanual.txt',
                    help='Test data csv file')
parser.add_argument('--num_train', type=int, required=False, default=1e8,
                    help='Maximum number of train data')
parser.add_argument('--num_val', type=int, required=False, default=1e8,
                    help='Maximum number of val data')
parser.add_argument('--num_test', type=int, required=False, default=1e8,
                    help='Maximum number of test data')
parser.add_argument('--seed', type=int, required=False, default=7240,
                    help='Random seed')

args = parser.parse_args()

random.seed(args.seed)


# Some miscellaneous functions
def check_dir_existence(path_dir):
    assert os.path.isdir(path_dir), \
        "Directory does not exist : {}".format(path_dir)


def check_file_existence(path_file):
    assert os.path.isfile(path_file), \
        "File does not exist : {}".format(path_file)


def main():
    check_dir_existence(args.path_root)
    check_dir_existence(args.path_out)

    assert (args.val_ratio >= 0.0) and (args.val_ratio <= 1.0), \
        "Validation set ratio should be in [0, 1] but {}".format(args.val_ratio)
    with open(args.txt_train, 'r') as f:
        files = f.readlines()
        files = [l[:-1] for l in files]
        files.sort()
        txt_train = files
    with open(args.txt_test, 'r') as f:
        files = f.readlines()
        files = [l[:-1] for l in files]
        files.sort()
        txt_test = files

    num_train = len(txt_train)
    num_test = len(txt_test)

    idx = np.arange(0, num_train)
    random.shuffle(idx)

    dict_json = {}

    num_val = int(num_train*args.val_ratio)
    num_train = num_train - num_val

    num_train = min(num_train, args.num_train)
    num_val = min(num_val, args.num_val)
    num_test = min(num_test, args.num_test)

    idx_train = idx[0:num_train]
    idx_val = idx[num_train:num_train+num_val]
    
    # root = "/data/fwei/scannet/ScanNet/SensReader/python/scannetv2_images"
    # Train
    list_data = []
    max_framenum = 100
    for i in idx_train:
        scene_name = txt_train[i]
        scene_frames = glob(f"{args.path_root}/{scene_name}/color/*")
        scene_frames.sort()
        if len(scene_frames) > max_framenum:
            scene_frames = random.sample(scene_frames, max_framenum)
        for frame in scene_frames:
            frame_name = frame.split('/')[-1].split('.')[0]
            path_rgb = f"{scene_name}/color/{frame_name}.jpg"
            path_depth = f"{scene_name}/depth/{frame_name}.png"
            path_gt = path_depth
            path_calib = f"{scene_name}/intrinsic/intrinsic_color.txt" 
            # path_calibd = f"{root}/{scene_name}/intrinsic/intrinsic_depth.txt" 

            dict_sample = {
                'rgb': path_rgb,
                'depth': path_depth,
                'gt': path_gt,
                'K': path_calib
            }

            flag_valid = True
            for val in dict_sample.values():
                flag_valid &= os.path.exists(args.path_root + '/' + val)
                if not flag_valid:
                    pause()

            list_data.append(dict_sample)

    dict_json['train'] = list_data

    print('Training data : {}'.format(len(list_data)))

    # Val
    list_data = []
    for i in idx_val:
        scene_name = txt_train[i]
        scene_frames = glob(f"{args.path_root}/{scene_name}/color/*")
        scene_frames.sort()
        if len(scene_frames) > max_framenum:
            scene_frames = random.sample(scene_frames, max_framenum)
        for frame in scene_frames:
            frame_name = frame.split('/')[-1].split('.')[0]
            path_rgb = f"{scene_name}/color/{frame_name}.jpg"
            path_depth = f"{scene_name}/depth/{frame_name}.png"
            path_gt = path_depth
            path_calib = f"{scene_name}/intrinsic/intrinsic_color.txt" 
            # path_calibd = f"{root}/{scene_name}/intrinsic/intrinsic_depth.txt" 

            dict_sample = {
                'rgb': path_rgb,
                'depth': path_depth,
                'gt': path_gt,
                'K': path_calib
            }

            flag_valid = True
            for val in dict_sample.values():
                flag_valid &= os.path.exists(args.path_root + '/' + val)
                if not flag_valid:
                    pause()

            list_data.append(dict_sample)

    dict_json['val'] = list_data

    print('Validation data : {}'.format(len(list_data)))

    # Test
    list_data = []
    for scene_name in txt_test:
        scene_frames = glob(f"{args.path_root}/{scene_name}/color/*")
        scene_frames.sort()
        if len(scene_frames) > max_framenum:
            scene_frames = random.sample(scene_frames, max_framenum)
        for frame in scene_frames:
            frame_name = frame.split('/')[-1].split('.')[0]
            path_rgb = f"{scene_name}/color/{frame_name}.jpg"
            path_depth = f"{scene_name}/depth/{frame_name}.png"
            path_gt = path_depth
            path_calib = f"{scene_name}/intrinsic/intrinsic_color.txt" 
            # path_calibd = f"{root}/{scene_name}/intrinsic/intrinsic_depth.txt" 

            dict_sample = {
                'rgb': path_rgb,
                'depth': path_depth,
                'gt': path_gt,
                'K': path_calib
            }

            flag_valid = True
            for val in dict_sample.values():
                flag_valid &= os.path.exists(args.path_root + '/' + val)
                if not flag_valid:
                    pause()

            list_data.append(dict_sample)

    dict_json['test'] = list_data

    print('Test data : {}'.format(len(list_data)))

    # Write to json files
    f = open(args.path_out + '/' + args.name_out, 'w')
    json.dump(dict_json, f, indent=4)
    f.close()

    print("Json file generation finished.")


if __name__ == '__main__':
    print('\nArguments :')
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    print('')

    main()
