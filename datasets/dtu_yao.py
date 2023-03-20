from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *

# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.pairfile = kwargs.get("pairfile", "pair.txt")

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]
        
        # scans
        pair_file = "Cameras/"+self.pairfile
        for scan in scans:            
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
            
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx)) # resolution 512x640 Nb: orig res. 1200x1600 
            mask_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_visual_{:0>4}.png'.format(scan, vid))                  # resolution 128x160 (.png)
            depth_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_map_{:0>4}.pfm'.format(scan, vid))                    # resolution 128x160  (.pfm)
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)                               # fx=361 (1/8th of original fx=2892) ?? 

            # Note on DTU intrinsics:
            # 1600x1200 with intrinsics (2892.3 0 823.2)
            # 640x512 with intrinsics (361.5 0.0 82.9)
            # step1: 1600x1200 truncated to 1280x1024 --> intrinsics (2892.3 0 663.2) 
            # step2: rows 1&2 divided by 8 --> intrinsics (361.5 0 82.9) 
            # but why 8 ?  1280 to 640 is a factor 2 only            
            
            imgs.append(self.read_img(img_filename))
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval, dtype=np.float32)
                mask = self.read_img(mask_filename)
                depth = self.read_depth(depth_filename)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        return {"imgs": imgs,
                "proj_matrices": proj_matrices,
                "depth": depth,
                "depth_values": depth_values,
                "mask": mask}


if __name__ == "__main__":
    
    # some testing code, just IGNORE it
    print ("## Dataset test - OLI## python -m datasets.dtu_yao")
    
    N = 3
    # datapath = "/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/DTU/mvs_training"
    datapath = "/home/deeplearning/BRO/EVAL_CODE/MVS/datasets/DTU/mvs_training"
    
    dataset = MVSDataset(datapath, '/home/deeplearning/BRO/EVAL_CODE/MVS/MVSNet_pytorch/lists/dtu/train.txt', 'train',N, 128)
    item = dataset[50]

    # dataset = MVSDataset("./MVS_TRANING", '/home/deeplearning/BRO/EVAL_CODE/MVS/MVSNet_pytorch/lists/dtu/val.txt', 'val', N,128)
    # item = dataset[50]

    # N = 5
    # dataset = MVSDataset("./MVS_TRANING", '/home/deeplearning/BRO/EVAL_CODE/MVS/MVSNet_pytorch/lists/dtu/test.txt', 'test', N, 128)
    # item = dataset[50]

    # test homography here
    print("dataset items (dict):", item.keys())
    print("imgs: ", item["imgs"].shape)
    print("depth: ", item["depth"].shape)
    print("depth_values: ", item["depth_values"].shape)
    print("mask", item["mask"].shape)

    print("\nImages:")
    ref_img = item["imgs"][0].transpose([1, 2, 0])[::4, ::4]
    src_imgs = [item["imgs"][i].transpose([1, 2, 0])[::4, ::4] for i in range(1, N)]
    print("ref_img shape: ", ref_img.shape)
    print("length src_imgs: ", len(src_imgs))
    
    ref_proj_mat = item["proj_matrices"][0]
    src_proj_mats = [item["proj_matrices"][i] for i in range(1, N)]
    print("ref_proj_mat: ", ref_proj_mat)
    
    mask = item["mask"]
    depth = item["depth"]

    height = ref_img.shape[0]
    width = ref_img.shape[1]
    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    print("yy", yy.max(), yy.min())
    yy = yy.reshape([-1])
    xx = xx.reshape([-1])
    X = np.vstack((xx, yy, np.ones_like(xx)))
    D = depth.reshape([-1])
    print("X", "D", X.shape, D.shape)

    X = np.vstack((X * D, np.ones_like(xx)))
    X = np.matmul(np.linalg.inv(ref_proj_mat), X)
    X = np.matmul(src_proj_mats[0], X)
    X /= X[2]
    X = X[:2]

    yy = X[0].reshape([height, width]).astype(np.float32)
    xx = X[1].reshape([height, width]).astype(np.float32)
    import cv2

    warped = cv2.remap(src_imgs[0], yy, xx, interpolation=cv2.INTER_LINEAR)
    warped[mask[:, :] < 0.5] = 0

    cv2.imwrite('../tmp0.png', ref_img[:, :, ::-1] * 255)
    cv2.imwrite('../tmp1.png', warped[:, :, ::-1] * 255)
    cv2.imwrite('../tmp2.png', src_imgs[0][:, :, ::-1] * 255)
