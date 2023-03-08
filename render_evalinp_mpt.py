from utils.util import *
from os.path import join
from pdb import set_trace as pause
import multiprocessing as mp
# fs_deppre = glob(dir_src+"/*/05_pred_final_gray.png")
# # fs_deppre = glob(dir_src+"/*/05_pred_final.png")
# fs_deppre.sort()
# fs_inp = sort_glob(f"{root_resultseg}/color")
# fs_mask = glob(f"{root_resultseg}/*/mask/*") #0")
# fs_mask.sort()
root_data = "/n/fs/rgbd/users/fwei/data/matterport/v1"
method = 'ff'
method = 'psr'
method = 'us'
method = 'spsg'
assert method in ['us', 'psr', 'ff', 'spsg']
# f_meshes = glob(f"{root_resultseg}/[0-7]*/poissonmeshes_depth10_3.5eval-recgt/*_hires.ply")
# f_meshes = glob(f"{root_resultseg}/[0-7]*/poissonmeshes_depth10_3.5-d.05-r.3/*_hires.ply")
fname = '/n/fs/rgbd/users/fwei/data/matterport/Matterport/tasks/benchmark/scenes_test.txt'
with open(fname, 'r') as f:
    files = f.readlines()
    files = [l[:-1] for l in files]
    files.sort()
house_ids = files
files = []
for hid in house_ids:
    if method == "us":
        f = join(root_data, hid, "poissonmeshes_depth10_3.5-[0-9]-*conf/*_hires.ply")
    elif method == "psr":
        f = join(root_data, hid, "poissonmeshes_depth10_3.5-psr*conf/*_hires.ply")
    elif method == "ff":
        result_root = "/home/fwei/project/prior_work/Free-form-3D-Scene-Inpainting/torch/output/test-c/vis"
        f = join(result_root, "*pred-mesh.ply")
    elif method == "spsg":
        result_root = "/home/fwei/project/prior_work/spsg/torch/output/test-c/vis"
        f = join(result_root, "*pred-mesh.ply")
    samples = glob(f)
    # t1 = len(glob(join(root, hid,"depth320-1000/*")))
    # t2 = len(glob(join(root, hid,"dephole320-1000/*")))
    # if t1!=t2:
    files += samples
files.sort()
pause()
# f_meshes = glob(f"{root_data}//poissonmeshes_depth10_3.5*conf/*_hires.ply")
# f_meshes.sort()
# for f_deppre, f_mask, f_depren in zip(fs_deppre, fs_mask, fs_rdep):
# for i, f_mesh in enumerate(f_meshes): #, fs_rimg):
scale = 4
img_dim = [320, 236]
def f(f_mesh):
    mesh_folder = f_mesh.split('/')[-2]
    house_root = f_mesh.split('/'+mesh_folder)[0]
    house_id = house_root.split('/')[-1]
    region = mesh_folder.split('-')[1]
    region_cnt = mesh_folder.split('-')[2]
    clutterf = glob(join(house_root, "clean_hole", f"region{region}-{region_cnt}*ply"))[0]
    frame_name = clutterf.split('-')[-1][:-4]+".jpg"
    dir_r = house_root + "/render_us"
    fimg = join(dir_r, frame_name)
    if os.path.exists(fimg):
        return
    f_conf = join(house_root, "undistorted_camera_parameters", f"{house_id}.conf")
    with open(f_conf, 'r') as f:
        lines = f.readlines()
    # n_img = int(lines[1].split()[1])
    # himg_list = [item.split()[2] for item in lines if item[:4]=="scan"]
    hcamext_dict = {item.split()[2]:list(map(float,item.split()[3:])) for item in lines if item[:4]=="scan"}
    hcamint_dict = {lines[i+1].split()[2][:-6]:list(map(float,item.split()[1:])) for i, item in enumerate(lines) if item[:4]=="intr"}
    intrinsic = np.array(hcamint_dict[frame_name[:-6]]).reshape(3,3)
    intrinsic[:2] /= scale
    pose = np.array(hcamext_dict[frame_name]).reshape((4,4))
    mesh = trimesh.load(f_mesh, process=False)
    meshp = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    # prepare scene
    r = pyrender.OffscreenRenderer(img_dim[0], img_dim[1])
    scene = pyrender.Scene(bg_color=[0, 0, 0]) #ambient_light=[s,s,s])
    scene.add(meshp)
    camera = pyrender.camera.IntrinsicsCamera(
            intrinsic[0, 0],intrinsic[1, 1],
            intrinsic[0, 2],intrinsic[1, 2])
    scene.add(camera, pose=pose)
    img, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    # scene.remove_node(list(scene.camera_nodes)[0])
    # scene.remove_node(list(scene.mesh_nodes)[0])
    if not os.path.exists(dir_r):
        try:
            os.makedirs(dir_r)
        except:
            pass
    imageio.imwrite(fimg, img)
    print(f_mesh,fimg)

# f(files[0])
# pause()
p = mp.Pool(processes=60) 
p.map(f, files) #[1:])
p.close()
p.join()
