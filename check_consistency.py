
from WarperPytorch import Warper
from utils.util import *
root_data = "/data/fwei/scannet/ScanNet/SensReader/python"
scene = "645"
exp_name = "221024_002501_test699small"
exp_name = "221102_175827_test699-2dl.3"
exp_name = "221102_175854_test645-2dl.3"
root_exp = f"/n/fs/rgbd/users/fwei/exp/clutter_bpnet/mink3.7/"
root_resultseg = f"{root_exp}/majc0_bnd1v3_csam_sizeb-04_2dl.3/result{scene}/best" #"/com_pre2/scene0699_00"
device="cuda:0"
# device="cpu"
fK = f"{root_data}/scannetv2_images/scene0{scene}_00/intrinsic/intrinsic_depth.txt"
K = read_2darray(fK)
K = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
K_ = [K[0]/K[2]*w0/2, K[1]/K[3]*h0/2, w0/2, h0/2]
K2 = [K[0]/K[2]*w/2, K[1]/K[3]*h/2, w/2, h/2]
K_ = [K[0]/640*w0, K[1]/480*h0, w0/2, h0/2]
K2 = [K[0]/640*w0, K[1]/480*h0, w/2, h/2]

remove_closer_deppre = True
remove_huge_mask = True
cross_frame = True
closer_list = []
mask_list = []
dir_src = f"/home/fwei/project/inpaint3d/experiments/{exp_name}/test/epoch0020"
fs_deppre = glob(dir_src+"/*/05_pred_final_gray.png")
# fs_deppre = glob(dir_src+"/*/05_pred_final.png")
fs_deppre.sort()
# fs_inp = sort_glob(f"{root_resultseg}/color")
fs_mask = sort_glob(f"{root_resultseg}/mask") #0")
fs_rimg = sort_glob(f"{root_resultseg}/render_img")
fs_rdep = sort_glob(f"{root_resultseg}/render_depth")
root_data = "/data/fwei/scannet/ScanNet/SensReader/python"
make_dir(f"{root_resultseg}/depth")
# make_dir(f"{root_resultseg}/color2")
# make_dir(f"{root_resultseg}/deppre_final")
all_poses = []
all_masks = []
all_depths = []
# for f_deppre, f_mask, f_depren in zip(fs_deppre, fs_mask, fs_rdep):
print("loading all masks & depths")
for f_deppre, f_mask in tqdm(zip(fs_deppre, fs_mask),total=len(fs_deppre)): #, fs_rimg):
    frame_name = get_frame_name(f_mask)
    frame_num = int(frame_name)
    fcam = f"{root_data}/scannetv2_images5/scene0{scene}_00/pose/{int(frame_name)}.txt"
    pose = read_2darray(fcam)
    all_poses.append(pose)
#     continue
    frame_num = int(frame_name)
#     deppre = imageio.imread(f_deppre)
    mask = imageio.imread(f_mask)#/255
    all_masks.append(mask)
    f_depcap = f"{root_data}/scannetv2_images5/scene0{scene}_00/depth/{int(frame_name)}.png"
    depcap = imageio.imread(f_depcap)/1000 #center_crop(imageio.imread(f_depcap), h, w)
    all_depths.append(depcap)

    # f_dst = f_inp.split("/color/")[0]+"/depth/"+frame_name.zfill(9)+".png"
    # f_dst = f_mask.split("/mask/")[0]+"/depth/"+frame_name.zfill(9)+".png"
    # # print(f_deppre,f_dst)
    # shutil.copy(f_deppre, f_dst)
    # f_dst = f_mask.split("/mask/")[0]+"/color/"+frame_name.zfill(9)+".jpg"
    # f_inp = f_mask.split("/mask/")[0]+"/inp/"+str(frame_num)+".jpg"
    # # f_inp = f_mask.replace("mask", "inp")
    # im = imageio.imread(f_inp)
    # im2 = center_crop(im, h, w)
    # # print(f_inp, f_dst)
    # imageio.imwrite(f_dst, im2)
all_masks = torch.from_numpy(np.stack(all_masks))[:, None]
all_depths = torch.from_numpy(np.stack(all_depths))[:, None]
all_images = all_depths.repeat(1,3,1,1)

remove_closer_deppre = True
remove_huge_mask = True
cross_frame0 = False 
cross_frame = True
closer_list = []
mask_list = []
# exp_name = "221024_002501_test699small"
# dir_src = f"/home/fwei/project/inpaint3d/experiments/{exp_name}/test/epoch0020"
# root_resultseg = f"/n/fs/rgbd/users/fwei/exp/clutter_bpnet/mink3.7/majc0_bnd1v3_csam_sizeb-04_csmal/result{scene}/com_pre2/scene0699_00"
fs_deppre = glob(dir_src+"/*/05_pred_final_gray.png")
# fs_deppre = glob(dir_src+"/*/05_pred_final.png")
fs_deppre.sort()
# fs_inp = sort_glob(f"{root_resultseg}/color")
fs_mask = sort_glob(f"{root_resultseg}/mask")
fs_rimg = sort_glob(f"{root_resultseg}/render_img")
fs_rdep = sort_glob(f"{root_resultseg}/render_depth")
root_data = "/data/fwei/scannet/ScanNet/SensReader/python"
make_dir(f"{root_resultseg}/depth")
make_dir(f"{root_resultseg}/depth_compre5")
make_dir(f"{root_resultseg}/depth_cfmask2.05")
make_dir(f"{root_resultseg}/depth_cfmask2mout.05")
# %reload_ext autoreload
# %autoreload 2
poses_t = torch.linalg.inv(torch.from_numpy(np.stack(all_poses)))
n_total = poses_t.shape[0]
masks_t = all_masks[:, :, 6:-6, 8:-8]
depths_t = all_depths[:, :,6:-6, 8:-8]
images_t = all_images[:, :,6:-6, 8:-8]
if cross_frame0:
    img = images_t.to(device)
    masks = masks_t.to(device)
    dep = depths_t.to(device)
    pose1s = poses_t.to(device)
if cross_frame:
    poses = poses_t.to(device)
fK = f"{root_data}/scannetv2_images/scene0{scene}_00/intrinsic/intrinsic_depth.txt"
K = read_2darray(fK)
K = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
K_ = [K[0]/640*w0, K[1]/480*h0, w0/2, h0/2]
K2 = [K[0]/640*w0, K[1]/480*h0, w/2, h/2]
intrinsic1 = np.array([[K2[0],0,K2[2]], [0, K2[1],K2[3]],[0, 0, 1]])
intrinsic1 = torch.from_numpy(intrinsic1)[None]
intrinsics = intrinsic1.expand(n_total,-1,-1).to(device)
depth_max = 1000
# for i, (f_deppre, f_mask, f_depren) in tqdm(enumerate(zip(fs_deppre, fs_mask, fs_rdep))):
for i, (f_deppre, f_mask) in tqdm(enumerate(zip(fs_deppre, fs_mask))): #, fs_rimg):
    frame_name = get_frame_name(f_mask)
    frame_num = int(frame_name)
    # if i<=220:continue
    if i>225:break
#     if frame_num != 670: continue #670
#     if i%5==0: print(i)
    fcam = f"{root_data}/scannetv2_images5/scene0{scene}_00/pose/{int(frame_name)}.txt"
    pose = read_2darray(fcam)

    deppre = imageio.imread(f_deppre)
    mask = center_crop(imageio.imread(f_mask),h,w)#/255
#     depren = center_crop(imageio.imread(f_depren), h, w)
#     imgren = imageio.imread(f_imgren)
    f_depcap = f"{root_data}/scannetv2_images5/scene0{scene}_00/depth/{int(frame_name)}.png"
    depcap = center_crop(imageio.imread(f_depcap), h, w)
    f_depcom = f"{root_resultseg}/depth/{frame_name.zfill(9)}.png"
    if (mask==0).sum()==0:
        f_depcom = f"{root_resultseg}/depth_compre5/{frame_name.zfill(9)}.png"
        shutil.copy(f_depcap, f_depcom)
        f_depcom = f"{root_resultseg}/depth_cfmask2.05/{frame_name.zfill(9)}.png"
        shutil.copy(f_depcap, f_depcom)
        f_depcom = f"{root_resultseg}/depth_cfmask2mout.05/{frame_name.zfill(9)}.png"
        shutil.copy(f_depcap, f_depcom)
        continue
    f_imginp = f"{root_resultseg}/color/{frame_name}.jpg"
    imginp = imageio.imread(f_imginp)
    f_imgcap = f"{root_data}/scannetv2_images5/scene0{scene}_00/color/{int(frame_name)}.jpg"
    imgcap = center_crop(imageio.imread(f_imgcap), h, w)
#     f_inpdst = f_inp.replace("/color/", "/color2/")
    # non-mask areas use captured depth
    depcom = deppre * (1-mask) + depcap * (mask)    
#     mask2 = (depcom.astype(float) - depren.astype(float) ) >= (depcap.astype(float) - depren.astype(float) ) -1
    # per pixel: inpainted depth >= captured depth
    mask2 = (depcom.astype(float)  ) >= (depcap.astype(float))# -1)
    depcom2 = depcom * mask2 + depcap * (1 - mask2)
    imginp = imginp * mask2[..., None] + imgcap * (1 - mask2)[..., None]
    # paste empty areas from captured depth to inpainted depth 
    # (because of ambiguity and large errors at boundaris betweeen different depths)
    depcom3 = depcom2 * (depcap!=0)
    depcom = depcom3
    # per masked component: mean(inpainted depth) >= mean(captured depth)
    # guaranteed non-zero pixels are the same 
    if remove_closer_deppre:
        mask_further_deppre = is_deppre_closer(mask, depcap, deppre)
        depcom4 = depcom3 * mask_further_deppre
        depcom = depcom4
        if (mask_further_deppre==0).sum()>0:
            closer_list.append(f_depcom)
            mask_list.append(mask_further_deppre)
    # if mask is larger than half of the image area: empty the masked depth.
    if remove_huge_mask and mask.sum()<(1-mask).sum():
        depcom5 = depcom * mask
    else:
        depcom5 = depcom * 1
    depcom = depcom5

#     f_depcom1 = f"{root_resultseg}/depth_cfmask2m/{frame_name.zfill(9)}.png"
#     f_depcom2 = f"{root_resultseg}/depth_cfmask2.05/{frame_name.zfill(9)}.png"
#     d1=imageio.imread(f_depcom1)
#     d2=imageio.imread(f_depcom2)
#     depcom=d1+(d2>0)*(d1<=0)*d2
#     f_depcom = f"{root_resultseg}/depth_cfmask2mout.05/{frame_name.zfill(9)}.png"
#     imageio.imwrite(f_depcom, (depcom).astype(np.uint16))

    f_depcom = f"{root_resultseg}/depth_compre5/{frame_name.zfill(9)}.png"
    imageio.imwrite(f_depcom, (depcom).astype(np.uint16))
#     break
    if cross_frame0:
        f_depcom = f"{root_resultseg}/depth_compre5/{frame_name.zfill(9)}.png"
        depcom5 = imageio.imread(f_depcom)
        depcom = depcom5/1000
        warper = Warper(device=f"gpu{device[-1]}" if len(device)>3 else "cpu")
        warped_frame2, warped_mask2, warped_depth2, flow12, mask3, _ = warper.forward_warp(img, masks, dep, pose1s, pose1s[i:(i+1)], intrinsics, None, render_image=False)
        mask3 *= warped_depth2>0
        mask_inp = 1 - masks_t[i:(i+1)]
        mask_int = mask_inp * mask3.cpu()
        depth_int = mask_int * warped_depth2.cpu()
        depth_int[depth_int==0] = depth_max
        depth_int[i] = depth_max
        depth_intmin = depth_int.min(0)[0].numpy()[0]
        depth_intmin[depth_intmin==depth_max] = 0
        mask_invalidinp = (depth_intmin > depcom) |  ((depth_intmin < depcom)&(depth_intmin > (dep[i,0].cpu().numpy()+0.05)))
        depcom6 = depcom * (1-mask_invalidinp)
        depcom = depcom6*1000
        del warper
        del warped_frame2, warped_mask2, warped_depth2, flow12, mask3
        gc.collect()
        torch.cuda.empty_cache()
    if cross_frame:
#         f_depcom = f"{root_resultseg}/depth_compre5/{frame_name.zfill(9)}.png"
#         depcom5 = imageio.imread(f_depcom)
        depcom = depcom5/1000.0
        warper = Warper(device=f"gpu{device[-1]}" if len(device)>3 else "cpu", memory_opt=True)#False)
        img = images_t[i].to(device).expand(n_total,-1,-1,-1)
        masks = masks_t[i].to(device).expand(n_total,-1,-1,-1)
        dep = torch.from_numpy(depcom[None]*1).to(device).expand(n_total,-1,-1,-1)
        pose1s = poses_t[i].to(device).expand(n_total,-1,-1)
        warped_frame2, warped_mask2, warped_depth2, flow12, mask3, id_vis = warper.forward_warp(img, 1-masks, dep, pose1s, poses, intrinsics, None, render_image=False)
#         mask_occ = ((warped_depth2 +0.03)< depths_t.to(device))&(warped_depth2>0)#& masks_t.to(device).bool()
#         mask_occ = ((warped_depth2)< depths_t.to(device))&(warped_depth2>0)#& masks_t.to(device).bool()
        mask_occ = ((warped_depth2+0.05) < depths_t.to(device))&(warped_depth2>0)#& masks_t.to(device).bool()
        dep2 = warped_depth2 * mask_occ
#         dep22 = warped_depth2 * (1-mask_occ.float())+mask_occ * depths_t.to(device)*(warped_depth2>0)
        warped_frame2, warped_mask1, warped_depth1, flow12, mask3, id_vis2 = warper.forward_warp(img, mask_occ, dep2, poses, pose1s, intrinsics, None, render_image=False)
        mask_invalidinp = mask3.sum(0).cpu().numpy()[0]>0
        depcom6 = depcom * (1-mask_invalidinp)*1000
        f_depcom = f"{root_resultseg}/depth_cfmask2.05/{frame_name.zfill(9)}.png"
        imageio.imwrite(f_depcom, (depcom6).astype(np.uint16))
#         break
        del dep2, mask_occ, id_vis2,warped_mask1, warped_depth1, mask3
#         del warped_frame2, warped_mask2, warped_depth2, flow12, mask3
#         del img, masks, dep, pose1s
        gc.collect()
        torch.cuda.empty_cache()
#     plt.imshow(mask_invalidinp.permute(1,2,0).numpy())
#     break
#     continue
        
        mask_occ = ((warped_depth2+0.05) < depths_t.to(device))&(warped_depth2>0) & masks_t.to(device).bool()
        dep2 = warped_depth2 * mask_occ
        warped_frame2, warped_mask1, warped_depth1, flow12, mask3, id_vis2 = warper.forward_warp(img, mask_occ, dep2, poses, pose1s, intrinsics, None, render_image=False)
        mask_invalidinp = mask3.sum(0).cpu().numpy()[0]>0
        depcom6 = depcom * (1-mask_invalidinp)
        depcom = depcom6*1000
        del warper, dep2, mask_occ, id_vis, id_vis2,warped_mask1, warped_depth1
        del warped_frame2, warped_mask2, warped_depth2, flow12, mask3
        del img, masks, dep, pose1s
        gc.collect()
        torch.cuda.empty_cache()
        f_depcom = f"{root_resultseg}/depth_cfmask2mout.05/{frame_name.zfill(9)}.png"
        imageio.imwrite(f_depcom, (depcom).astype(np.uint16))
    print(f_depcom)
    break
#     imageio.imwrite(f_inpdst, imginp.astype(np.uint8))
if cross_frame0:
    del img, masks, dep, pose1s
    gc.collect()
    torch.cuda.empty_cache()
if cross_frame0 or cross_frame:
    del intrinsics,poses
    gc.collect()
    torch.cuda.empty_cache()
