import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from datasets import find_dataset_def
from models import *
from utils import *
import sys
import re
from datasets.data_io import read_pfm, save_pfm
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse. May be different from the original implementation')

parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='dtu_yao_eval', help='select dataset')
parser.add_argument('--testpath', help='testing data path')
parser.add_argument('--testlist', help='testing scan list')
parser.add_argument('--pairfile', default="pair.txt", help='pair filename')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--NviewGen', type=int, default=5, help='number of views used to generate depth maps (DTU=5)')
parser.add_argument('--NviewFilter', type=int, default=10, help='number of src views used while filtering depth maps (DTU=10)')
parser.add_argument('--photomask', type=float, default=0.8, help='photometric confidence mask: pixels with photo confidence below threshold are dismissed')
parser.add_argument('--geomask', type=int, default=3, help='geometric view mask: pixels not seen by at least a certain number of views are dismissed ')
parser.add_argument('--condmask_pixel', type=float, default=1.0, help='conditional mask pixel: pixels which reproject back into the ref view at more than the threshold number of pixels are dismissed')
parser.add_argument('--condmask_depth', type=float, default=0.01, help='conditional mask on relative depth difference: pixels with depths prediction values above a threshold (1%) are dismissed')

parser.add_argument('--debug_MVSnet', type=int, default=0, help='powers of 2 for switches selection (debug = 2⁰+2¹+2³+2⁴+...) with '
                    '0: print matrices and plot features (add 1) '
                    '1: plot warped views (add 2) '
                    '2: plot regularization (add 4) '
                    '3: plot depths proba (add 8) '
                    '4: plot expectation (add 16) '
                    '5: plot photometric confidence (add 32) ')
parser.add_argument('--debug_depth_gen', type=int, default=0, help='powers of 2 for switches selection (debug = 2⁰+2¹+2³+2⁴+...) with '
                    '0: plot input image (add 1) '
                    '1: plot depth predictions and resp. confidence for each cam (add 2) '
                    '3: plot depth with masks (add 4)'
                    )


# Check line280 for img filename format

def get_powers(n):
    return [str(p) for p,v in enumerate(bin(n)[:1:-1]) if int(v)]

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    
    return intrinsics, extrinsics

# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img

# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5

# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool_
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)

# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            data.append((ref_view, src_views))
    return data

#======SAVE_DEPTH============================================================================================================

# run MVS model to save depth maps and confidence maps
def save_depth():
    # N_GenViews = 2 # OLI (Min:2, default: 5) depth inference only using the ref view and first nviews-1 src views from pair_file.txt
    print("============ Generating DEPTH MAPS using {} views".format(args.NviewGen)) # OLI
    
    # dataloader
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(datapath=args.testpath, 
                              listfile=args.testlist, 
                              mode="test", 
                              nviews=args.NviewGen, 
                              ndepths=args.numdepth, 
                              interval_scale=args.interval_scale, 
                              pairfile=args.pairfile) 
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=10, drop_last=False)
    
    # model
    model = MVSNet(refine=False, debug=args.debug_MVSnet)
    model = nn.DataParallel(model)
    model.cuda()

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
    model.eval()
    # print(model) # OLI

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader): 
            # info:  sample => dict_keys(['imgs', 'proj_matrices', 'depth_values', 'filename']) with sample["imgs"].shape=torch.Size([1, 2, 3, 1184, 1600])=[B, Nimg, RGB, H, W]
            
            #  DEBUG
            if '0' in get_powers(args.debug_depth_gen): # add 1
                print("## [DEBUG] save_depth, batch_idx: {}".format(batch_idx)) 
                # print ("sample keys: ",sample.keys())
                for iview in range(args.NviewGen):
                    BRG_img = sample["imgs"].permute(3,4,2,0,1)[::2,::2,:,0,iview].numpy()
                    RGB_img = cv2.cvtColor(BRG_img, cv2.COLOR_BGR2RGB)
                    cv2.imshow('view:{} batch:{} Res.:{}'.format(iview, batch_idx, str(RGB_img.shape)), RGB_img) # OLI     
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # END DEBUG 
            
            sample_cuda = tocuda(sample)
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            outputs = tensor2numpy(outputs)
            del sample_cuda
            print('Iter {}/{}'.format(batch_idx, len(TestImgLoader)))
            filenames = sample["filename"]

            # save depth maps and confidence maps
            for filename, depth_est, photometric_confidence in zip(filenames, outputs["depth"], outputs["photometric_confidence"]):    
                
                # folder filenames
                acquisition_folder = args.testpath.split('/')[-1]
                depth_filename = os.path.join(args.outdir, acquisition_folder, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, acquisition_folder, filename.format('confidence', '.pfm'))
                
                # create folders 'depth_est' & 'confidence'
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                
                # save depth maps
                save_pfm(depth_filename, depth_est)
                print("PFM saved: {}".format(depth_filename))
                
                # save confidence maps
                save_pfm(confidence_filename, photometric_confidence)

                #  DEBUG
                if '1' in get_powers(args.debug_depth_gen): # add 2
                    depth_esti_norm = (depth_est - np.min(depth_est)) / (np.max(depth_est) - np.min(depth_est))
                    
                    print ("depth res.: ", depth_est.shape)
                    print ("depth Min/Max: ",np.min(depth_est), np.max(depth_est))
                    print ("conf. Min/Max: ",np.min(photometric_confidence), np.max(photometric_confidence))
                    
                    cv2.imshow("[depth estim.] view:{} res.:{}".format(batch_idx, str(depth_esti_norm.shape)), np.uint8(depth_esti_norm * 255)) 
                    cv2.imshow("[confidence] view:{}".format(batch_idx), np.uint8(photometric_confidence * 255)) 
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    


#########FILTER_DEPTH#####################################################################################################3

# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    """Oli info:
        step 1: project ref_img points into src_img frame
        step 2: interpolate src_img depths (available on src_img points) onto projected ref_img points
        step 3: project back the interpolated src_img depths (now available on the projected ref_img points) onto the ref_img frame        
    """
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    
    ## step1. project reference pixels to the source view
    ##########
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1]) # flatten
    # reference 3D space 
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space (Oli: projection of xyz_ref into src view)
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    ##########
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):

    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height)) # (296, 400), (296, 400)
    
    # Oli: reproject depth info from src_img onto ref_img to evaluate conditions for consistency 
    print("depth reprojection..", end="") # OLI
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                                                                depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2) # (296, 400)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    # ================ CONDITIONAL MASK ================
    # Oli: Establish mask based on conditions: projected_pts_dist < 1 pixel & depth_diff < 1%         
    mask = np.logical_and(dist < args.condmask_pixel, relative_depth_diff < args.condmask_depth) # (296, 400) default / DTU
    
    print("conditional mask: {}/{} ({:.2f}%)".format(mask.sum(), mask.shape[0]*mask.shape[1], 100*mask.sum()/(mask.shape[0]*mask.shape[1])))
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(dataset_folder, scan, out_folder, plyfilename):
    print("===== FILTER DEPTHs =====") # OLI
    print("Dataset:{}\n",format(dataset_folder))
    
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    # Read pair file
    pair_file = os.path.join(dataset_folder, "Cameras", args.pairfile)
    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)
    print("Reading pair files:\n{}\n".format(pair_data))
    
    # TODO: hardcode size
    # used_mask = [np.zeros([296, 400], dtype=np.bool) for _ in range(nviews)]

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        print ("=> Ref view: {}, SRC views: {}".format(ref_view, src_views)) # OLI
        
        # load the camera parameters for REFERENCE VIEW
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            # os.path.join(dataset_folder, 'cams/{:0>8}_cam.txt'.format(ref_view))) # DTU testing
            os.path.join(dataset_folder, 'Cameras/{:0>8}_cam.txt'.format(ref_view))) # unified Camera path
        
        # RESCALE intrinsics: assume the feature is 1/4 of the original image size
        ref_intrinsics[:2, :] /= 4.0  ### input/output factor from CNN 
        
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]  # (296, 400)
        
        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0] # (296, 400)
        
        # load the reference image        
        # ref_img = read_img(os.path.join(dataset_folder, 'images/{:0>8}.jpg'.format(ref_view))) # Orig
        # ref_img = read_img(os.path.join(dataset_folder, 'Rectified/{}/{:0>8}.jpg'.format(scan, ref_view))) #OLI -  DTU  
        # ref_img = read_img(os.path.join(dataset_folder, 'Rectified/{}/{:0>8}.png'.format(scan, ref_view))) #OLI - Merlin  
        ref_img = read_img(os.path.join(dataset_folder, 'Rectified_raw/{}/rect_{:0>3}_3_r5000.png'.format(scan, ref_view+1))) # 1200x1600
        ref_img_resized = ref_img[0::4, 0::4, :] # img reduced to resolution of predicted depth using the IO factor from CNN
        h_depth, w_depth = ref_depth_est.shape
        ref_img_cropped = ref_img_resized[0:h_depth, 0:w_depth, :] 
        
        print("confidence percentiles: 25%:{:.1f}% 50%:{:.1f}% 75%:{:.1f}% 90%:{:.1f}%".format(np.percentile(confidence, 25)*100, 
                                                                                            np.percentile(confidence, 50)*100, 
                                                                                            np.percentile(confidence, 75)*100, 
                                                                                            np.percentile(confidence, 90)*100) )

        # ================== PHOTO MASK =================== 
        # compute the photo MASK 
        photo_mask = confidence > args.photomask # DTU
        # photo_mask = confidence > 0 # Blender
   
        # Initialize variable for conditinoal mask
        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # compute the geometric MASK  
        geo_mask_sum = 0
        for src_view in src_views[:args.NviewFilter]: # only use the first NviewFilter views of pairfile.txt
        # for src_view in src_views: # filter depth using all src views from pair_file.txt (for each ref view)
            
            # camera parameters of the SOURCE VIEW
            src_intrinsics, src_extrinsics = read_camera_parameters(
                # os.path.join(dataset_folder, 'Cameras/{}/{:0>8}_cam.txt'.format(scan, src_view))) # orig
                os.path.join(dataset_folder, 'Cameras/{:0>8}_cam.txt'.format(src_view)))
            
            # RESCALE intrinsics: assume the feature is 1/4 of the original image size
            src_intrinsics[:2, :] /= 4.0           
            
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]
            
            print("SRC view...", end="") # OLI
            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                                        src_depth_est, src_intrinsics, src_extrinsics) # (296, 400)
            
            geo_mask_sum += geo_mask.astype(np.int32) # (296, 400) with int values 0,1,2... (adding mask of each src_view)
            all_srcview_depth_ests.append(depth_reprojected) # list of arrays of shape (296, 400)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        # compute average of depth prediction over all views
        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        
        # ================ geo MASK =====================
        #  at least 3 source views matched  
        geo_mask = geo_mask_sum >= args.geomask # DTU, info Oli: only take pixels which have been included in at least 3 masks
        
        
        # ==== FILTER ==== combination of all masks
        final_mask = np.logical_and(photo_mask, geo_mask) # oli info: only take pixels which belong to both  masks
        

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("SUMMARY: Ref_view: {:0>2}, photo/geo/final-mask:{:.2f}%/{:.2f}%/{:.2f}%\n".format(ref_view,
                                                                                                     photo_mask.mean()*100,
                                                                                                     geo_mask.mean()*100, 
                                                                                                     final_mask.mean()*100))

        #  DEBUG: plot depth with masks
        if '2' in get_powers(args.debug_depth_gen): # add 4
            
            img_norm = cv2.cvtColor(ref_img_cropped, cv2.COLOR_BGR2RGB)
            cv2.imshow('ref_img', img_norm)

            ref_depth_est_norm = (ref_depth_est - np.min(ref_depth_est)) / (np.max(ref_depth_est)-np.min(ref_depth_est)) 
            cv2.imshow('ref_depth', ref_depth_est_norm)
            cv2.imshow('photo_mask', ref_depth_est_norm * photo_mask.astype(np.float32) )
            cv2.imshow('geo_mask', ref_depth_est_norm * geo_mask.astype(np.float32) )
            cv2.imshow('final mask', ref_depth_est_norm * final_mask.astype(np.float32) )
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        # BUILD 3D points (vertices) to be kept (appended) for a given ref_img
        height, width = depth_est_averaged.shape[:2] 
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
        valid_points = final_mask
        print("3D Pts-cloud: No of valid_points: {}/{} (average={:03f})".format(valid_points.sum(), height*width, valid_points.mean())) 
        
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        # color = ref_img[0:-16:4, 0::4, :][valid_points]  # hardcoded for DTU dataset, images cropped by 16pixels at bottom, see dtu_yao_eval.py
        color = ref_img_cropped[valid_points]  # scaling and cropping done above
        
        # color = ref_img[1::4, 1::4, :][valid_points]  # hardcoded for Merlin dataset
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics), np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics), np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8)) # list of arrays containing vertices (x,y,z)

        # # set used_mask[ref_view]
        # used_mask[ref_view][...] = True
        # for idx, src_view in enumerate(src_views):
        #     src_mask = np.logical_and(final_mask, all_srcview_geomask[idx])
        #     src_y = all_srcview_y[idx].astype(np.int)
        #     src_x = all_srcview_x[idx].astype(np.int)
        #     used_mask[src_view][src_y[src_mask], src_x[src_mask]] = True

        #  DEBUG: plot 3D point-cloud
        if '3' in get_powers(args.debug_depth_gen): # add 8
            
            import open3d as o3d
            # Create frame and point cloud
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_world.transpose((1, 0)).astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(color)
            # plot (type h for help in open3d)
            o3d.visualization.draw_geometries([frame]+[pcd], front=[0.8,0.13,-0.6],lookat=[40.1,33.4,595],up=[-0.42,-0.57,-0.70],zoom=0.38)

    #  Once all reference images processed, concatenate all vertices and save as ply format
    vertexs_xyz = np.concatenate(vertexs, axis=0)
    vertexs_xyz_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs_xyz], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertexs_xyz_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    
    print("saving the final model to", plyfilename)
    
            
    #  DEBUG: plot FINAL 3D point-cloud
    if '4' in get_powers(args.debug_depth_gen): # add 16
        
        import open3d as o3d
        # Create frame and point cloud
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertexs_xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(vertexs_xyz_colors/255)
        # plot
        o3d.visualization.draw_geometries([frame]+[pcd], front=[0.8,0.13,-0.6],lookat=[40.1,33.4,595],up=[-0.42,-0.57,-0.70],zoom=0.38)

if __name__ == '__main__':

    # step1. save all the depth maps and the masks in outputs directory
    #  nb: depth inference the ref view and first nviews-1 src views from pair_file.txt
    # save_depth()

    with open(args.testlist) as f:
        scans = f.readlines()
        scans = [line.rstrip() for line in scans] 

    for scan in scans:
        # scan_id = int(scan[4:]) # if using "scanXXX"
        scan_id = int(re.findall(r'\d+', scan)[0])
        # dataset_folder = os.path.join(args.testpath, scan)
        dataset_folder = os.path.join(args.testpath)
        acquisition_folder = args.testpath.split("/")[-1]
        out_folder = os.path.join(args.outdir, acquisition_folder, scan)
        
        # step2. filter saved depth maps with photometric confidence maps and geometric constraints
        # NviewFilter  (Min:1, Max: as per pair.txt file) only use limited number of SOURCE views from pair.txt 
        
        plyfilename = os.path.join(args.outdir, acquisition_folder, 'mvsnet{:0>3}_l3.ply'.format(scan_id))
        filter_depth(dataset_folder, scan, out_folder, plyfilename ) 
