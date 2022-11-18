# %matplotlib inline
import cv2
from matplotlib import pyplot as plt
import numpy as np
import shutil
import scipy
import skimage
import copy
import os
import os.path as osp
# import pandas as pd
from glob import glob
import imageio.v2 as imageio
import math
# from ply import *
import torch
import torch.nn.functional as F
# %load_ext autoreload
from tqdm import tqdm
from skimage.measure import label
import gc
import trimesh
# import plyfile
# import sklearn}
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.system('PYOPENGL_PLATFORM=egl python -c "from OpenGL import EGL"') 
import pyrender
epsilon=1e-10

#import open3d as o3d; print(o3d.__version__)
def center_crop(image, h, w):
    center = image.shape
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2
    crop_img = image[int(y):int(y+h), int(x):int(x+w)]
    return crop_img

def sq2ori(imsq, imori, l):
    if imori is None:
        imori = np.ones((h,w))
    img_sq = cv2.resize(imgsq,(240,240))
    img_inp = imori * 1
    img_inp[:,l:(l+240)]=img_sq
    return img_inp

def read_2darray(fcam):
    with open(fcam, 'r') as f:
        lines = f.readlines()
        lines = [list(map(float, line.split())) for line in lines] 
        pose = np.array(lines)
    return pose
# root = "/Users/weifangyin/Downloads/tt/tmp"

def depth2mesh(K, fK=None, depth=None, depthPath=None, rgb=None, rgbPath=None, meshPath=None, 
               #mtlPath, matName, useMaterial = True,
               #K = [286.70755254580064, 286.70031481696736], 
               pose=None, 
               fcam= None,
               pyrender_depth=True
              ):
               #"/Users/weifangyin/Downloads/tt/tmp/t8/00000001/cam.txt"):
    if depth is None:
        depth = imageio.imread(depthPath).astype(np.float32) 
    if depth.max() > 10:
        depth = depth / 1000.0
    #depth = np.load(f"{root_result}/00000001/depth_ori2.npy")
    h, w = depth.shape
    if rgb is None:
        try:
            rgbPath = os.path.dirname(depthPath) + "/01_rgb.png"
            rgb = imageio.imread(rgbPath).astype(np.float32) / 255.0
            assert rgb.shape == (h, w, 3)
        except:
            rgb = np.ones((h,w,3))
            print("no color saved!")
    if pose is None:
        pose = read_2darray(fcam)
#     if max(meshPath.find('\\'), meshPath.find('/')) > -1:
#         os.makedirs(os.path.dirname(mashPath), exist_ok=True)

    ids = np.zeros((depth.shape[1], depth.shape[0]), int)
    vid = 0 #1
    v_pos = []
    v_rgb = []
    f_idx = []
    for u in range(0, w):
        for v in range(h-1, -1, -1):

            d = depth[v, u]

            ids[u,v] = vid
            if d == 0.0:
                ids[u,v] = 0
            vid += 1

            x = u - w/2 + pyrender_depth
            y = v - h/2 + pyrender_depth
            x = x*d / K[0]
            y = d*y / K[1]
            z = d        
            drgb = rgb[v, u] #[::-1]

            v_cam = np.array([x,y,z,1])
            v_wld = pose @ v_cam.T
            v_pos.append(v_wld[:3])
            v_rgb.append(drgb[:3])
            #         for u in range(0, depth.shape[1]):
#             for v in range(0, depth.shape[0]):
#                 f.write("vt " + str(u/depth.shape[1]) + " " + str(v/depth.shape[0]) + "\n")
    for u in range(0, depth.shape[1]-1):
        for v in range(0, depth.shape[0]-1):

            v1 = ids[u,v]; v2 = ids[u+1,v]; v3 = ids[u,v+1]; v4 = ids[u+1,v+1];
            if v1 == 0 or v2 == 0 or v3 == 0 or v4 == 0:
                continue
            f_idx.append([v2, v1, v3])
            f_idx.append([v2, v3, v4])
#                 f.write("f " + vete(v1,v1) + " " + vete(v2,v2) + " " + vete(v3,v3) + "\n")
#                 f.write("f " + vete(v3,v3) + " " + vete(v2,v2) + " " + vete(v4,v4) + "\n")
    mesh = trimesh.Trimesh(vertices=np.stack(v_pos,0), faces=np.stack(f_idx,0), vertex_colors=np.stack(v_rgb,0))
    if meshPath is not None:
        tt=mesh.export(meshPath)
    return mesh

h,w=228,304
h0, w0 = 240,320
def get_inside_mask(v_ori, K_):
    v = v_ori * 1.0
    v[0] = v[0] * K_[0] / v[2] + K_[2]
    v[1] = v[1] * K_[1] / v[2] + K_[3]
    vr = np.round(v-0.5).astype(int)[:2]
    inside_mask = (vr[0]>=0)*(vr[1]>=0)*(vr[0]<w0)*(vr[1]<h0)*(v[2]>0)
    return inside_mask,vr

def trans_verts(meshori, pose_inv):
    ret=meshori.copy()
    ret.apply_transform(pose_inv)
    return ret
    v_camori = np.concatenate([meshori.vertices,np.ones((meshori.vertices.shape[0],1))],1)
    v_camori = pose_inv @ v_camori.T
    mesh = trimesh.Trimesh(vertices=v_camori[:3].T, faces=meshori.faces) #, vertex_colors=meshori.)
    return mesh#, v_camori

def mesh2depth(mesh,pose_inv,K, w0=w0,h0=h0):
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.system('PYOPENGL_PLATFORM=egl python -c "from OpenGL import EGL"') 
    import pyrender
    r = pyrender.OffscreenRenderer(w0,h0) 
    scene = pyrender.Scene(bg_color=[0, 0, 0])
    camera = pyrender.camera.IntrinsicsCamera(K[0]/K[2]*w0/2, K[1]/K[3]*h0/2, w0/2, h0/2)
    cam_pose = np.eye(4)
    cam_pose[1, 1]=-1
    cam_pose[2, 2]=-1
    scene.add(camera, pose=cam_pose)
    #facesn = np.stack([meshori.faces[:,1],meshori.faces[:,0],meshori.faces[:,2]],-1)
    #meshtrans = trimesh.Trimesh(vertices=v_camori[:3].T, faces=facesn) 
#     meshtrans = trimesh.Trimesh(vertices=v_camori[:3].T, faces=meshori.faces)
    meshtrans = trans_verts(mesh, pose_inv)
    mesh = pyrender.Mesh.from_trimesh(meshtrans)
    scene.add(mesh)
    img, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT\
                          |pyrender.constants.RenderFlags.SKIP_CULL_FACES)
    return img, depth

#clean mesh_fillarea
def remove_length_edges(mesh_fillarea, edge_thresh = 0.02):
    mesh_fillarea.edges_face.shape,mesh_fillarea.edges.shape
    edge_len = mesh_fillarea.vertices[mesh_fillarea.edges]
    edge_len = ((edge_len[:,0] - edge_len[:,1])**2).sum(1)**0.5
    edge_badidx = edge_len > edge_thresh
    face_badidx = mesh_fillarea.edges_face
    face_badidx=np.unique(mesh_fillarea.edges_face[edge_badidx])
    face_goodmask = np.ones(mesh_fillarea.faces.shape[0]).astype(bool)
    face_goodmask[face_badidx] = 0
    meshfill_good=mesh_fillarea.copy()
    meshfill_good.update_faces(face_goodmask)
    meshfill_good.remove_unreferenced_vertices()
    return meshfill_good

def stats(x):
    print(f"min: {x.min()}, max: {x.max()}, mean: {x.mean():.6f}, sum: {x.sum():.6f}, dtype: {x.dtype}, shape: {x.shape}")
    
def pltshow(fn):
    im = imageio.imread(fn)
    plt.imshow(im)
    print(fn)
    return im

# import shutil
# for i in seg_files:
#     shutil.move(i, i.replace('best/mrgb_', 'mrgb/'))
def sort_glob(path):
    from glob import glob
    files = glob(path+"/*")
    files.sort()
    files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0])) 
    return files


def get_frame_name(path):
    frame_name = path.split("/")[-1].split(".")[0]
    return frame_name


def get_scene_name(path):
    scene_name = path.split("/")[-3]
    return scene_name


def make_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
def is_deppre_closer(mask, depcap, deppre):
    mask_ret = np.ones_like(mask)
    mask_label = label(mask,background=1)
    for ilabel in np.unique(mask_label)[1:]:
        idepcap = depcap * (mask_label == ilabel)    
        ideppre = deppre * (mask_label == ilabel)
        mask = (mask_label == ilabel)  *(idepcap>0)*(ideppre>0)
        idepcap *= mask
        ideppre *= mask
        cap_mean = idepcap.sum()/(1e-9+(idepcap!=0).sum())
        pre_mean = ideppre.sum()/(1e-9+(ideppre!=0).sum())
#         print(ilabel,cap_mean,pre_mean,np.unique(idepcap))
        if depcap.max()>10:
            buffer = 5
        else:
            buffer = 0.005
        if cap_mean > pre_mean-buffer:
            mask_ret[mask_label == ilabel] = 0
    return mask_ret


