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
import open3d as o3d

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse. May be different from the original implementation')

parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='dtu_yao', choices=['dtu_yao', 'blender', 'dataloader_eval'],help='select dataloader file')
parser.add_argument('--dataset_name', default='dtu', choices=['dtu', 'bds1', 'bds2', 'bds4', 'bds6', 'bds7', 'bin'],help='select dataset type')
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
                    '5: plot photometric confidence (add 32) '
                    '63: ALL'    
                    )

parser.add_argument('--debug_depth_gen', type=int, default=0, help='powers of 2 for switches selection (debug = 2⁰+2¹+2³+2⁴+...) with '
                    '0: plot input image (add 1) '
                    '1: plot depth predictions and confidence for each cam (add 2) '
                    '2: plot 3D point-cloud for each view (add 4)'
                    '3: plot combined 3D point-cloud (add 8)'
                    '4:  (add 16)'
                    )

parser.add_argument('--debug_depth_filter', type=int, default=0, help='powers of 2 for switches selection (debug = 2⁰+2¹+2³+2⁴+...) with '
                    '0: plot depth with masks (add 1)'
                    '1: plot 3D point-cloud for each view (add 2)'
                    '2: plot fused 3D point-cloud (add 4)'
                    '7: ALL'
                    )

# Check line280 for img filename format

def get_powers(n):
    return [str(p) for p,v in enumerate(bin(n)[:1:-1]) if int(v)]

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

def NormalizeNumpy(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def read_camera_parameters(filename):
    """ reads intrinsics and extrinsics """
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    
    # RESCALE intrinsics: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4.0  ## DISABLED when reading cam parameters from the output folder
    
    return intrinsics, extrinsics


def write_cam(file, K, R, depth_params):    
    """ reads intrinsics and extrinsics in MVS format"""
    f = open(file, "w")
    
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(R[i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(K[i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(depth_params[0]) + ' ' + str(depth_params[1]) + ' ' + str(depth_params[2]) + ' ' + str(depth_params[3]) + '\n')
    
    f.close()
    

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

# Get Open3D box
def get_o3d_frame_bbox(dims=(0.57, 0.37, 0.22), delta = (0,0,0), scale = 1, context=None):
    """generate Open3D object for bin and frame

    Args:
        dims (tuple, optional): Bin box dimensions (in m). Defaults to (0.54, 0.34, 0.2).
        delta (tuple, optional): Offset to apply to bin box. Mostly for non-Blender (i.e real datasets) where world ref is determined empirically. Defaults to (0,0,0).
        scale (int, optional): scale factor. Defaults to 1.

    Returns:
        Open3D objects:frame, bounding_box, bounding_box2 (bbox with 2cm wall offset) - WARNING: in mm
    """
    
    # Test Configurations (overides other arguments if defined)
    if context is not None:
        if "overhead03" in context:
            # dims = (0.54, 0.34, 0.2)
            dims = (0.57, 0.37, 0.22)
            delta = (0.08, 0.03, .0)
        elif "overhead02" in context:
            # dims = (0.54, 0.34, 0.2)
            dims = (0.57, 0.37, 0.22)
            delta = (0.08, 0.03, .0)
        else:
            dims = (0.57, 0.37, 0.22)
            delta = (0,0,0)
                
    # Plot axis and 3D points in WORLD ref
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100*scale, origin=[0, 0, 0])

    # Change dimensions from m to mm
    dims_bin = np.asarray(dims)*1000
    delta_orig = np.asarray(delta)*1000
    
    # Apply scaling factor if any    
    dims_bin *= scale
    delta_orig *= scale
    
    # Create Bounding box for bin internal walls
    min_bbox = -dims_bin / 2.0
    max_bbox = dims_bin / 2.0
    max_bbox[2] -= min_bbox[2]
    min_bbox[2] = 0 
    
    bbox = o3d.geometry.AxisAlignedBoundingBox()
    bbox.min_bound = min_bbox + delta_orig
    bbox.max_bound = max_bbox + delta_orig
    bbox.color = [0, 0, 1]
    
    # Create Bounding box for bin external walls
    wall_size = 20   # in mm
    
    bbox2 = o3d.geometry.AxisAlignedBoundingBox()
    bbox2.min_bound = min_bbox + delta_orig - np.array((wall_size, wall_size, wall_size))
    bbox2.max_bound = max_bbox + delta_orig + np.array((wall_size, wall_size, 0))
    bbox2.color = [1, 0, 0]
    
    # o3d.visualization.draw_geometries([frame] + [bbox] + [bbox2])
    
    return frame, bbox, bbox2 

# Helper
def invert(rotation_translation):
    '''Invert a 3D rotation matrix in their (R | t) representation    '''
    rot = rotation_translation[0].T
    trans = -rot @ rotation_translation[1]
    return (rot, trans)


# Get camera for Open3D representation   
def get_o3d_cameras(cam_extrinsics, highlight_1st=False):
    
    cams=[]
    for i, extrinsics in enumerate(cam_extrinsics):
        rotation = extrinsics[:3,:3]
        translation = extrinsics[:3,-1]

        camera_rotation, camera_translation = invert((rotation, translation))

        height = 30
        cam = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=5,cone_radius=10,cylinder_height=height,cone_height=height/3)
        cam.compute_vertex_normals()
        if i==0 and highlight_1st:
            cam.paint_uniform_color((1,0,0))
        else:
            cam.paint_uniform_color((0,0,1))
        cam.translate([0, 0, -height])
        cam.rotate(camera_rotation, center=np.array([0, 0, 0]))
        cam.translate(camera_translation)
        
        cams.append(cam)
    
    return cams


def depth2pts_np(depth_map, cam_intrinsic, cam_extrinsic):
    feature_grid = get_pixel_grids_np(depth_map.shape[0], depth_map.shape[1])

    uv = np.matmul(np.linalg.inv(cam_intrinsic), feature_grid)
    cam_points = uv * np.reshape(depth_map, (1, -1))

    R = cam_extrinsic[:3, :3]
    t = cam_extrinsic[:3, 3:4]
    R_inv = np.linalg.inv(R)

    world_points = np.matmul(R_inv, cam_points - t).transpose()
    return world_points

def get_pixel_grids_np(height, width):
    x_linspace = np.linspace(0.5, width - 0.5, width)
    y_linspace = np.linspace(0.5, height - 0.5, height)
    x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)
    x_coordinates = np.reshape(x_coordinates, (1, -1))
    y_coordinates = np.reshape(y_coordinates, (1, -1))
    ones = np.ones_like(x_coordinates).astype(np.float)
    grid = np.concatenate([x_coordinates, y_coordinates, ones], axis=0)

    return grid


###### SAVE_DEPTH #############################################################################################################
################################################################################################################################

# run MVS model to save depth maps and confidence maps
def save_depth(cam_subfolder, img_subfolder,img_res):
    
    # NviewGen (Min:2, default: 5) depth inference only using the ref view and first nviews-1 src views from pair_file.txt
    print("============ Generating DEPTH MAPS using {} views".format(args.NviewGen))
    print ("cam_subfolder: ",cam_subfolder)
    print ("img_subfolder: ",img_subfolder)
    print ("img_res: ",img_res)
    
    # dataloader
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(datapath=args.testpath, 
                              listfile=args.testlist, 
                              mode="test", 
                              nviews=args.NviewGen, 
                              ndepths=args.numdepth, 
                              interval_scale=args.interval_scale, 
                              pairfile=args.pairfile,
                              cam_subfolder = cam_subfolder,
                              img_subfolder = img_subfolder,
                              img_res = img_res, 
                              dataset_name = args.dataset_name) 
    
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

    # Final point cloud 
    vertices = []
    vertices_colors = []
    cam_extrinsics = []
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):   # note: batch MUST be 1 for eval
                       
            # sample info:  
            # sample => dict_keys(['imgs', 'proj_matrices', 'depth_values', 'filename', 'intrinsics', 'extrinsics']) 
            #    with sample["imgs"].shape=torch.Size([1, 2, 3, 1184, 1600])=[B, Nimg, RGB, H, W]
            
            #  DEBUG - Plot input image + img channels
            if '0' in get_powers(args.debug_depth_gen): # add 1                
                print("## [DEBUG] (save_depth) batch_idx: {}".format(batch_idx))                 
                for iview in range(args.NviewGen):
                    BRG_img = sample["imgs"].permute(3,4,2,0,1)[::2,::2,:,0,iview].numpy()
                    RGB_img = cv2.cvtColor(BRG_img, cv2.COLOR_BGR2RGB)
                    cv2.imshow('[EVAL] View:{} B:{} HalfRes.:{}'.format(iview, batch_idx, str(RGB_img.shape)), RGB_img) # OLI     
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # END DEBUG 
            
            # Save ref image            
            filenames = sample["filename"]
            acquisition_folder = args.testpath.split('/')[-1]
            img_filename = os.path.join(args.outdir, acquisition_folder, filenames[0].format("images", ".png"))
            os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
            BRG_img = sample["imgs"].permute(3,4,2,0,1)[:,:,:,0,0].numpy()
            RGB_img = cv2.cvtColor(BRG_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_filename, np.uint8(RGB_img * 255))
            
            # Get cam parameters
            intrinsics_list = sample["intrinsics"] # list of tensors
            extrinsics_list = sample["extrinsics"]
                
            # Model Fwd pass for prediction
            timestamp = time.time()
            sample_cuda = tocuda(sample)
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            outputs = tensor2numpy(outputs)
            del sample_cuda
            print(f'Iter {batch_idx+1}/{len(TestImgLoader)} (fwd pass in {round(time.time()-timestamp,2)}s)')


            # save depth maps and confidence maps
            for ite, (filename, depth_est, photometric_confidence) in enumerate(zip(filenames, outputs["depth"], outputs["photometric_confidence"])):    

                # get camera parameters
                intrinsics = intrinsics_list[ite].squeeze().numpy()
                extrinsics = extrinsics_list[ite].squeeze().numpy()
                if ite ==0:
                    ref_intrinsics = intrinsics
                    ref_extrinsics = extrinsics
                    cam_extrinsics.append(ref_extrinsics)   
                                                  
                # create folder filenames
                depth_filename = os.path.join(args.outdir, acquisition_folder, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, acquisition_folder, filename.format('confidence', '.pfm'))
                cam_filename = os.path.join(args.outdir, acquisition_folder, filename.format('cams', '_cam.txt'))
                
                # create folders 'depth_est' & 'confidence'
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                
                # save depth maps
                save_pfm(depth_filename, depth_est)
                cv2.imwrite(depth_filename.replace(".pfm",".png"), np.uint8(NormalizeNumpy(depth_est)*255))
                print("PFM saved: {}".format(depth_filename))
                
                # save confidence maps
                save_pfm(confidence_filename, photometric_confidence)
                cv2.imwrite(confidence_filename.replace(".pfm",".png"), photometric_confidence)
                
                # Save cams
                write_cam(cam_filename, K=intrinsics, R=extrinsics, depth_params=["000", "2.5","",""])
        
                # print info
                print("depth Min/Max: {:.1f}/{:.1f} - conf. Min/Max: {:.1f}%/{:.1f}%".format(np.min(depth_est), 
                                                                                        np.max(depth_est),
                                                                                        np.min(photometric_confidence)*100, 
                                                                                        np.max(photometric_confidence)*100) )
                      
                print("confidence percentiles: 25%:{:.1f}% 50%:{:.1f}% 75%:{:.1f}% 90%:{:.1f}%".format(np.percentile(photometric_confidence, 25)*100, 
                                                                                                        np.percentile(photometric_confidence, 50)*100, 
                                                                                                        np.percentile(photometric_confidence, 75)*100, 
                                                                                                        np.percentile(photometric_confidence, 90)*100) )
                
                print("Depth & confidence filed saved to: {}\n".format(depth_filename))
                
                
                #  DEBUG - Plot depth estimation
                if '1' in get_powers(args.debug_depth_gen): # add 2                    
                    
                    print(f"[depth_gen] intrinsics:\n{ref_intrinsics}")
                    
                    depth_esti_norm = (depth_est - np.min(depth_est)) / (np.max(depth_est) - np.min(depth_est))                                      
                    cv2.imshow("[depth estim.] view:{} res.:{}".format(batch_idx, str(depth_esti_norm.shape)), depth_esti_norm ) 
                    cv2.imshow("[confidence] view:{}".format(batch_idx), photometric_confidence ) 
                    
                    mask = (photometric_confidence>0.5)
                    p2 = photometric_confidence.copy()
                    p2[~mask] = 0
                    
                    cv2.imshow("[confidence >50%] view:{}".format(batch_idx), np.uint8(p2 * 255)) 
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                #  DEBUG END
                    
                    
                # Save 3D points cloud                
                xyz_world = depth2pts_np(depth_est, ref_intrinsics, ref_extrinsics) # all points from depthmap included
                h, w = depth_est.shape
                ref_img = sample["imgs"][0,0].permute(1,2,0).numpy()
                ref_img_rescaled = cv2.resize(ref_img, (w, h))
                xyz_color = ref_img_rescaled.reshape(-1,3)
                
                # Add to global point cloud
                vertices.append(xyz_world)
                vertices_colors.append(xyz_color)
                    
                    
                #  DEBUG - Plot 3D point cloud for each view
                if '2' in get_powers(args.debug_depth_gen): # add 4
                                        
                    # Create frame and bounding boxes
                    frame, bbox, bbox2 = get_o3d_frame_bbox(context = acquisition_folder)
                    
                    # get_camera objects
                    o3D_cameras = get_o3d_cameras([ref_extrinsics], True)
                    
                    # Create  point cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz_world)
                    pcd.colors = o3d.utility.Vector3dVector(xyz_color)
                    pcd.estimate_normals()
                    
                    if args.dataset_name == "dtu":
                        o3d.visualization.draw_geometries([frame]+[pcd]+o3D_cameras)
                    else:
                        o3d.visualization.draw_geometries([frame]+[bbox]+[bbox2]+[pcd]+o3D_cameras)
                #  DEBUG END
                    
                    
        #  Once all images processed, concatenate all vertices and save as ply format
        print("Combining ALL 3D Pts-clouds.\n")
        vertices_allviews = np.concatenate(vertices, axis=0)
        vertices_colors_allviews = np.concatenate(vertices_colors, axis=0)
        
        
        # DEBUG - Plot final combined 3D point cloud
        if '3' in get_powers(args.debug_depth_gen): # add 8
            print("plotting combined 3D point clouds.")  
                    
            # Create frame and bounding boxes
            frame, bbox, bbox2 = get_o3d_frame_bbox(context=acquisition_folder)
                
            # get_camera objects
            o3D_cameras = get_o3d_cameras(cam_extrinsics, False)
            
            # Create  point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices_allviews)
            pcd.colors = o3d.utility.Vector3dVector(vertices_colors_allviews)
            pcd.estimate_normals()
            
            # Vizu
            if args.dataset_name == "dtu":
                o3d.visualization.draw_geometries([frame]+[pcd]+o3D_cameras)
            else:
                o3d.visualization.draw_geometries([frame]+[bbox]+[bbox2]+[pcd]+o3D_cameras)
                        
            # Vizu           
            pcd = pcd.crop(bbox2) 
            pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio = 2.0)
            pcd = pcd.voxel_down_sample(voxel_size=5)
            o3d.visualization.draw_geometries([frame]+[bbox]+[bbox2]+[pcd]+o3D_cameras)
        #  DEBUG END

           

#########FILTER_DEPTH##########################################################################################################
################################################################################################################################


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
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref), np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    
    # source 3D space (Oli: projection of xyz_ref into src view)
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)), np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    
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
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src), np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)), np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


###############################################################################################################################################

def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):

    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height)) 
    
    # Oli: reproject depth info from src_img onto ref_img to evaluate conditions for consistency 
    print("depth reprojection..", end="") # OLI
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                                                                depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2) 

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    # ================ CONDITIONAL MASK ================
    # Oli: Establish mask based on conditions: projected_pts_dist < 1 pixel & depth_diff < 1%         
    mask = np.logical_and(dist < args.condmask_pixel, relative_depth_diff < args.condmask_depth) 
    
    print("conditional mask: {}/{} ({:.2f}%)".format(mask.sum(), mask.shape[0]*mask.shape[1], 100*mask.sum()/(mask.shape[0]*mask.shape[1])))
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


###############################################################################################################################################

def filter_depth(dataset_folder, 
                 scan, 
                 out_folder, 
                 plyfilename,
                 cam_subfolder,
                 img_subfolder,
                 img_res,
                 ):
    
    print("============ DEPTH MAPS FILTER / FUSION using {} views".format(args.NviewFilter))
    print("Dataset:{}\n".format(dataset_folder))
    
    # for the final point cloud
    vertexs = []
    vertex_colors = []
    cam_extrinsics = []
    
    # Read pair file
    # pair_file = os.path.join(dataset_folder, cam_subfolder, args.pairfile)
    if args.dataset_name == "dtu" and cam_subfolder == "Cameras/train":
        pair_file = os.path.join(dataset_folder, cam_subfolder, "..", args.pairfile)  
    else:
        pair_file = os.path.join(dataset_folder, cam_subfolder, args.pairfile)  
        
    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)
    print("Reading pair files:\n{}\n".format(pair_data))
    
    # TODO: hardcode size
    # used_mask = [np.zeros([296, 400], dtype=np.bool) for _ in range(nviews)]

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        
        print ("=> Ref view: {}, SRC views: {}".format(ref_view, src_views)) # OLI
        
        # REFERENCE VIEW - Filenames - OBSOLETE
        # cam_filename =  os.path.join(dataset_folder, cam_subfolder, '{:0>8}_cam.txt'.format(ref_view))    # read from dataset folder
        # if args.dataset_name in ["dtu"]: 
        #     img_filename = os.path.join(dataset_folder, img_subfolder.format(scan, ref_view+1)) 
        # else:            
        #     img_filename = os.path.join(dataset_folder, img_subfolder.format(scan, ref_view))
        
        # REFERENCE VIEW - Filenames
        cam_filename =  os.path.join(args.outdir, args.testpath.split('/')[-1], scan, "cams", "00000{:0>3}_cam.txt".format(ref_view)) # better read from output folder in case image & cams were rescaled  
        img_filename =  os.path.join(args.outdir, args.testpath.split('/')[-1], scan, "images", "00000{:0>3}.png".format(ref_view)) # read from output folder (saved during depth generation)  
        depth_filename = os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view))
        conf_filename = os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view))
        
        # Read cam parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(cam_filename) 
        cam_extrinsics.append(ref_extrinsics)
        
        # Read depth prediction
        ref_depth_est = read_pfm(depth_filename)[0]  # (296, 400)
        h_d, w_d = ref_depth_est.shape[:2]
        
        # Read photometric confidence        
        confidence = read_pfm(conf_filename)[0] # (296, 400)
        
        # Read image  
        ref_img = read_img(img_filename)
        h_i, w_i = ref_img.shape[:2]
        
        # check
        assert (h_i, w_i) == (4*h_d, 4*w_d) , "incompatible depth and image dimensions."
        
        # Generate image at reduced depth dimensions 
        ref_img_resized = cv2.resize(ref_img, (w_d, h_d))           # img reduced to resolution of predicted depth using the IO factor from CNN
        
        # Print summary
        print("confidence percentiles: 25%:{:.1f}% 50%:{:.1f}% 75%:{:.1f}% 90%:{:.1f}%".format(np.percentile(confidence, 25)*100, 
                                                                                            np.percentile(confidence, 50)*100, 
                                                                                            np.percentile(confidence, 75)*100, 
                                                                                            np.percentile(confidence, 90)*100) )

        # ================== PHOTO MASK =================== 
        # compute the photo MASK 
        photo_mask = confidence > args.photomask # DTU
   
        # Initialize variable for conditinoal mask
        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # compute the geometric MASK  
        geo_mask_sum = 0
        
        # for src_view in src_views: # filter depth using all src views from pair_file.txt (for each ref view)
        for counter, src_view in enumerate(src_views[:args.NviewFilter]): 
            
            print("SRC view...", end="") 
            
            # SOURCE VIEW - Filenames
            # cam_filename = os.path.join(dataset_folder, cam_subfolder, '{:0>8}_cam.txt'.format(src_view)) # Obsolete
            cam_filename =  os.path.join(args.outdir, args.testpath.split('/')[-1], scan, "cams", "00000{:0>3}_cam.txt".format(src_view)) # read from output folder in case cams were rescaled  
            depth_filename = os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view))
            
            # Read cam parameters
            src_intrinsics, src_extrinsics = read_camera_parameters(cam_filename)
            
            # Read depth prediction
            src_depth_est = read_pfm(depth_filename)[0]
            
            # check geometric consistency
            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                                        src_depth_est, src_intrinsics, src_extrinsics) 
            
            # Build geometry mask
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

        print("SUMMARY: Ref_view: {:0>2}, photo/geo/final-mask:{:.2f}%/{:.2f}%/{:.2f}%".format(ref_view,
                                                                                                     photo_mask.mean()*100,
                                                                                                     geo_mask.mean()*100, 
                                                                                                     final_mask.mean()*100))

        #  DEBUG: plot depth with masks
        if '0' in get_powers(args.debug_depth_filter): # add 1
            
            img_norm = cv2.cvtColor(ref_img_resized, cv2.COLOR_BGR2RGB)
            cv2.imshow('ref_img', img_norm)

            ref_depth_est_norm = (ref_depth_est - np.min(ref_depth_est)) / (np.max(ref_depth_est)-np.min(ref_depth_est)) 
            cv2.imshow('ref_depth', ref_depth_est_norm)
            cv2.imshow('photo_mask', ref_depth_est_norm * photo_mask.astype(np.float32) )
            cv2.imshow('geo_mask', ref_depth_est_norm * geo_mask.astype(np.float32) )
            cv2.imshow('final mask', ref_depth_est_norm * final_mask.astype(np.float32) )
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # DEBUG END

        # BUILD 3D points (vertices) to be kept (appended) for a given ref_img
        height, width = depth_est_averaged.shape[:2] 
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
        valid_points = final_mask
        print("3D Pts-cloud: Number of valid_points: {}/{} (average={:03f})\n".format(valid_points.sum(), height*width, valid_points.mean())) 
        
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        # color = ref_img[0:-16:4, 0::4, :][valid_points]  # hardcoded for DTU dataset, images cropped by 16pixels at bottom, see dtu_yao_eval.py
        color = ref_img_resized[valid_points]  # scaling and cropping done above
        
        # color = ref_img[1::4, 1::4, :][valid_points]  # hardcoded for Merlin dataset
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics), np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics), np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8)) # list of arrays containing vertices (x,y,z)


        #  DEBUG: plot 3D point-cloud for each view
        if '1' in get_powers(args.debug_depth_filter): # add 2
            
            # Create frame and bounding boxes 
            frame, bbox, bbox2 = get_o3d_frame_bbox(context = img_filename)
            
            # get_camera objects
            o3D_cameras = get_o3d_cameras([ref_extrinsics], True)
            
            # Create  point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_world.transpose((1, 0)).astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(color)
            pcd.estimate_normals()
            
            # plot (type h for help in open3d)
            if args.dataset_name in ["dtu"]:
                o3d.visualization.draw_geometries([frame]+[pcd]+o3D_cameras, front=[0.8,0.13,-0.6],lookat=[40.1,33.4,595],up=[-0.42,-0.57,-0.70],zoom=0.38) # DTU
            else:
                o3d.visualization.draw_geometries([frame]+[bbox]+[bbox2]+[pcd]+o3D_cameras) # Blender, Bin
        # DEBUG END


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
    if '2' in get_powers(args.debug_depth_filter): # add 4

        # Create  point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertexs_xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(vertexs_xyz_colors/255)
        pcd.estimate_normals()
        
        # Create frame and bounding boxes
        frame, bbox, bbox2 = get_o3d_frame_bbox(context = img_filename)
            
        # get_camera objects
        o3D_cameras = get_o3d_cameras(cam_extrinsics, False)
            
        # plot (type h for help in open3d)
        if args.dataset in ["dtu"]:
            # Set view
            front, lookat, up, zoom = [0.8,0.13,-0.6], [40.1,33.4,595], [-0.42,-0.57,-0.70], 0.38
            o3d.visualization.draw_geometries([frame]+[pcd]+o3D_cameras, front=front, lookat=lookat, up=up, zoom=zoom) 
        else:
            # Set view
            # front, lookat, up, zoom = [-0.54,-0.52,0.66 ], [6.6,40.9,47.1], [0.52,0.42,0.75], 0.6                       
            
            # Vizu
            o3d.visualization.draw_geometries([frame]+[bbox]+[bbox2]+[pcd]+o3D_cameras)
            
            # Down-sample     
            pcd = pcd.crop(bbox2)
            pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio = 2.0)
            pcd = pcd.voxel_down_sample(voxel_size=5)
            o3d.visualization.draw_geometries([frame]+[bbox]+[bbox2]+[pcd]+o3D_cameras)

        
################################################################################################################################
################################################################################################################################

if __name__ == '__main__':
 
    # step0: get images, cameras SUBFOLDERS based on dataset name
    dict_cam_subfolder = {  "dtu": "Cameras",
                            "bds1": "Cameras_1200x1600",
                            "bds2": "Cameras_512x640",
                            "bds4": "Cameras_1024x1280",
                            # "bds4": "Cameras_512x640",
                            "bds6": "Cameras_1024x1280",
                            "bds7": "Cameras_512x640",
                            "bin": "Cameras",
                        }
    
    dict_img_subfolder = {  "dtu": "Rectified_raw/{}/rect_{:0>3}_3_r5000.png",
                            "bds1": "Rectified_1200x1600/{}/rect_C{:0>3}_L00.png",
                            "bds2": "Rectified_512x640/{}/rect_C{:0>3}_L00.png",
                            "bds4": "Rectified_1024x1280/{}/rect_C{:0>3}_L00.png",
                            # "bds4": "Rectified_512x640/{}/rect_C{:0>3}_L00.png",
                            "bds6": "Rectified_1024x1280/{}/rect_C{:0>3}_L00.png",
                            "bds7": "Rectified_512x640/{}/rect_C{:0>3}_L00.png",
                            "bin": "Rectified/{}/00000{:0>3}.png",
                        }
    

    dict_img_res = {"dtu": (600, 800),
                    # "dtu": (1200, 1600),
                    "bds1": (1200, 1600),
                    "bds2": (512, 640),
                    "bds4": (1024, 1280),
                    # "bds4": (512, 640),
                    "bds6": (1024, 1280),
                    "bds7": (512, 640),
                    "bin": (512, 640), # works
                    # "bin": (576, 800), # works
                    # "bin": (672, 800), # no works
                    # "bin": (1024, 1280),
                    # "bin": (1024, 1536),
                    # "bin": (672, 800),
                    # "bin": (1280, 1600),
                    # "bin": (1152, 1600),
                    # "bin": (2048, 2048),
                    # "bin": (2048, 3072),
                    }
        
    
     
    # step1. save all the depth maps and the masks in outputs directory
    #  nb: depth inference the ref view and first nviews-1 src views from pair_file.txt
    save_depth(cam_subfolder = dict_cam_subfolder[args.dataset_name],
               img_subfolder = dict_img_subfolder[args.dataset_name],
               img_res = dict_img_res[args.dataset_name],
               )

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
        filter_depth(dataset_folder, 
                     scan, 
                     out_folder, 
                     plyfilename,
                     dict_cam_subfolder[args.dataset_name],
                     dict_img_subfolder[args.dataset_name],
                     dict_img_res[args.dataset_name]
                     ) 
