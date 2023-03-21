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
        self.raw = kwargs.get("raw", True)

        assert self.mode == "test"
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # pair_file = "{}/pair.txt".format(scan)
        pair_file = "Cameras/"+self.pairfile
        
        for scan in scans: # scans is a list of subfolders: ['scan1', 'scan4', ...]
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    metas.append((scan, ref_view, src_views))
        print("[DataLoader] Mode:{} Nviews_pairfile:{} #scenes:{} #metas:{} ".format(self.mode, num_viewpoint, len(scans), len(metas)))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
            
        # read extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # read intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] /= 4 # input to feature CNN factor
        
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        
        if self.raw:
            assert np_img.shape[:2] == (1200, 1600)             
            np_img = np_img[:-16, :]  # crop to (1184, 1600) , do not need to modify intrinsics if cropping the bottom part
        else:
            assert np_img.shape[:2] == (512, 640) 
        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            # img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid)) 
            # proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))
            
            # Store image filenames
            if self.raw:
                img_filename = os.path.join(self.datapath, 'Rectified_raw/{}/rect_{:0>3}_3_r5000.png'.format(scan, vid+1)) 
            else:
                if "DTU" in self.datapath:
                    img_filename = os.path.join(self.datapath, 'Rectified/{}_train/rect_{:0>3}_3_r5000.png'.format(scan, vid+1)) 
                elif "Blender" in self.datapath:
                    img_filename = os.path.join(self.datapath, 'Rectified/{}/rect_{:0>3}_3_r5000.png'.format(scan, vid+1)) 
            print ("[dataloader] img fname:", img_filename)
            imgs.append(self.read_img(img_filename))
                
            # Read camera parameters
            # low res img (512x640) --> depth (128x160)
            # high res img (1184x1600) --> depth (296x400)
            # both have I/O CNN factor of 4 
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/{:0>8}_cam.txt'.format(vid))
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            if not self.raw:
                intrinsics[0,:] /= 2.5
                intrinsics[1,:] /= 2.3125

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval, dtype=np.float32)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        return {"imgs": imgs,
                "proj_matrices": proj_matrices,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}


if __name__ == "__main__":
    # some testing code, just IGNORE it
    dataset = MVSDataset("../dataset/mvs_testing/dtu/", '../lists/dtu/test.txt', 'test', 5, 128)
    item = dataset[50]
    for key, value in item.items():
        print(key, type(value))
