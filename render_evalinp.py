from utils.util import *
from pdb import set_trace as pause
import multiprocessing as mp
root_resultseg = f"/n/fs/rgbd/users/fwei/exp/clutter_bpnet/mink3.7/majc0_bnd1v3_csam_sizeb1_2dl.3/result_evalinp/best"
# fs_deppre = glob(dir_src+"/*/05_pred_final_gray.png")
# # fs_deppre = glob(dir_src+"/*/05_pred_final.png")
# fs_deppre.sort()
# fs_inp = sort_glob(f"{root_resultseg}/color")
fs_mask = glob(f"{root_resultseg}/*/mask/*") #0")
fs_mask.sort()
root_data = "/data/fwei/scannet/ScanNet/SensReader/python"
f_meshes = glob(f"{root_resultseg}/[0-7]*/poissonmeshes_depth10_3.5eval-psr/*_hires-hf.ply")
f_meshes.sort()
# for f_deppre, f_mask, f_depren in zip(fs_deppre, fs_mask, fs_rdep):
# for i, f_mesh in enumerate(f_meshes): #, fs_rimg):
def f(f_mesh):
    root_data = "/data/fwei/scannet/ScanNet/SensReader/python"
    frame_names = glob(f"{f_mesh.split('poisson')[0]}mask/*")
    frame_names.sort()
    f_mask = frame_names[len(frame_names)//2]
    scene = get_scene_name(f_mask)
    fK = f"{root_data}/scannetv2_images/scene0{scene}/intrinsic/intrinsic_depth.txt"
    K = read_2darray(fK)
    K = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    K_ = [K[0]/640*w0, K[1]/480*h0, w0/2, h0/2]
    frame_name = get_frame_name(f_mask)
    fcam = f"{root_data}/scannetv2_images5/scene0{scene}/pose/{int(frame_name)}.txt"
    pose = read_2darray(fcam)
    pose[1:3]*=-1
    pose_inv = np.linalg.inv(pose)
    mesh = trimesh.load(f_mesh, process=False)
    im_ori,depth=mesh2depth(mesh, pose_inv,K_,w0=w0,h0=h0)
    dir_hf = f"{f_mesh.split('poisson')[0]}evalinp"
    make_dir(dir_hf)
    imageio.imwrite(f"{dir_hf}/hf-{frame_name}.png", im_ori)
    print(dir_hf)
p = mp.Pool(processes=30) 
p.map(f, f_meshes) 
p.close()
p.join()
