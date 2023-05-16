from torch.utils.data import Dataset
import numpy as np
import os
import math
from PIL import Image
from datasets.data_io import read_pfm, save_pfm, read_rescale_crop_img


DEBUG = False # local debugging

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
        self.cam_subfolder = kwargs.get("cam_subfolder", True)
        self.img_subfolder = kwargs.get("img_subfolder", True)
        self.img_res = kwargs.get("img_res", True)
        self.dataset_name = kwargs.get("dataset_name", "dtu")

        assert self.mode == "test"
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        pair_file = os.path.join(self.cam_subfolder, self.pairfile)  
        
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
        
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        
        # Read
        img = Image.open(filename)
                     
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        
        # checks shape
        assert np_img.shape[:2] == self.img_res
        
        # check image has 3 channels (RGB), stack if only 1 channel
        if len(np_img.shape) == 2:
            np_img = np.dstack((np_img, np_img, np_img))
                    
        # Check resolution is multiple of 32, crop if needed
        # EX: (512,640) --> (512,640)
        # EX: (1024,1280) --> (1024,1280)
        # EX: (1200,1600) --> (1184, 1600) !!
        new_h, new_w  = tuple(i // 32 * 32 for i in self.img_res)
        np_img = np_img[:new_h, :new_w]  # cropping at bottom or right does not need intrinsics change
            
        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def __getitem__(self, idx):
        
        meta = self.metas[idx]
        if DEBUG: print("[dataloader] idx: ", idx)
        
        scan, ref_view, src_views = meta
        if DEBUG: print ("[dataloader] scan, ref_view, src_views: ", scan, ref_view, src_views )
        
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]
        if DEBUG: print ("[dataloader] view_ids: ",view_ids)
        
        # Init
        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []
        intrinsics_list = []
        extrinsics_list = []

        for i, vid in enumerate(view_ids):            
            
            # Filenames
            if self.dataset_name in ["dtu"]: 
                img_vid = vid + 1     
            else:
                img_vid = vid       
            img_filename = os.path.join(self.datapath, self.img_subfolder.format(scan, img_vid)) 
            proj_mat_filename = os.path.join(self.datapath, self.cam_subfolder,'{:0>8}_cam.txt'.format(vid))            
                
            # Read RAW camera parameters
            if DEBUG: print ("[dataloader] cam_filename: ", proj_mat_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            if DEBUG: 
                print (f"[dataloader] RAW intrinsics:\n{intrinsics}")
                print (f"[dataloader] extrinsics:\n{extrinsics}")

            # Read image, transform if needed and store
            if DEBUG: print ("[dataloader] img fname:", img_filename)
            # img = self.read_img(img_filename)
            # np_img, intrinsics = self.read_rescale_crop_img(img_filename, intrinsics)  
            np_img, intrinsics = read_rescale_crop_img(img_filename, intrinsics, img_res=self.img_res, DEBUG=DEBUG)   
            imgs.append(np_img)           

            # Resize intrinsics AFTER image/intrinsics adjustment to match CNN feature scale factor (& depth resolution)
            # rescale intrinsics by factor 4
                # low res img (512x640) --> depth (128x160) 
                # mid res img (1024x1280) --> depth (256x320) 
                # high res img (1184x1600) --> depth (296x400)
                # ALL have I/O CNN factor of 4 
            intrinsics[:2, :] /= 4.0 
            # magic_factor = 0.75
            # intrinsics[0,0] *= 1
            # intrinsics[1,1] *= magic_factor
            if DEBUG: print (f"[dataloader] FINAL intrinsics:\n{intrinsics}")
            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)
        
            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval, dtype=np.float32)
                if DEBUG: 
                    print ("[dataloader] number of depth values:", len(depth_values))
                    print ("[dataloader] Min/Max depth values:", min(depth_values), max(depth_values))

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        return {"imgs": imgs,                       # (B, C, H, W)
                "proj_matrices": proj_matrices,     # (4, 4, 4)
                "intrinsics": intrinsics_list,
                "extrinsics": extrinsics_list,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}


if __name__ == "__main__":
    # some testing code, just IGNORE it
    dataset = MVSDataset("../dataset/mvs_testing/dtu/", '../lists/dtu/test.txt', 'test', 5, 128)
    item = dataset[50]
    for key, value in item.items():
        print(key, type(value))
