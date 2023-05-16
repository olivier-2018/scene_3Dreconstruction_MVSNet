import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *


def get_powers(n):
    return [str(p) for p,v in enumerate(bin(n)[:1:-1]) if int(v)]

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32 

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1) # (in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x


class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2) # (in_channels, out_channels, kernel_size=3, stride=1, pad=1)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class MVSNet(nn.Module):
    def __init__(self, refine=True, debug=0):
        super(MVSNet, self).__init__()
        self.refine = refine
        self.debug = debug
        print('[MVSNet] init (debug={})'.format(self.debug))

        self.feature = FeatureNet()
        self.cost_regularization = CostRegNet()
        if self.refine:
            self.refine_network = RefineNet()

    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1) # list of NtrainViews tensors [B, RGBch, 512, 640]
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # DEBUG: plot input image
        if "0" in get_powers(self.debug):
            import cv2
            # plot images features
            for view in range(len(imgs)): # sweep through views (ref view + src views)
                cv2.imshow('[IMG] View:{} RGB img'.format(view), imgs[view].permute(2,3,1,0)[:,:,:,0].cpu().detach().numpy())
                for RGBchannel in range(0,imgs[view].shape[1],1): # show all 3 RGB channels sparately
                    cv2.imshow('[IMG] View:{} RGBchannel:{}'.format(view, RGBchannel), imgs[view].permute(2,3,1,0)[:,:,RGBchannel,0].cpu().detach().numpy())
                cv2.waitKey(0)
            cv2.destroyAllWindows()
        # DEBUG END
        
        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs] # list of NtrainViews tensors [B, 32, 128, 160]
        ref_feature, src_features = features[0], features[1:]  
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        
        # DEBUG: print and plot FEATURES
        if "0" in get_powers(self.debug):
            import cv2, re
            # plot images features
            for nview in range(len(features)): # sweep through views
                for filter in range(0,features[nview].shape[1],4): # select filter every 4 
                    cv2.imshow('[FEATURES]view:{} feat:{}'.format(nview, filter), features[nview].permute(2,3,1,0)[:,:,filter,0].cpu().detach().numpy())
                cv2.waitKey(0)
            cv2.destroyAllWindows()
            # print Transform. matrices
            for nview in range(len(features)):
                print("Matrix ref: {}\n".format(nview))
                print(re.sub('( \[|\[|\])', '',str(proj_matrices[nview].cpu().numpy())))
        # DEBUG END

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1) # tensor [1, 32, 128, 128, 160]
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2  # tensor [1, 32, 128, 128, 160]
        del ref_volume
        
        counter = 0 # OLI
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values) # tensor [1, 32, 128, 128, 160]
            
            # DEBUG: plot warped views
            if '1' in get_powers(self.debug):      
                import cv2
                # plot images features
                for filter in range(0,warped_volume.shape[1],8): # sweep through filters every 8
                    for depth in range(0,warped_volume.shape[2],12): # sweep through depths every 12
                        cv2.imshow('[WARPED-FEAT] Vpair:{} D:{}, F:{}'.format(counter,depth, filter), warped_volume.permute(3,4,1,2,0)[:,:,filter,depth].cpu().detach().numpy())
                    cv2.waitKey(0)
                cv2.destroyAllWindows()
                counter += 1
            # DEBUG END           
             
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
            
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # step 3. cost volume regularization
        cost_reg = self.cost_regularization(volume_variance)
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        
        # DEBUG: plot regularization
        if '2' in get_powers(self.debug):
            import cv2
            for depth in range(0,cost_reg.shape[2],16): # sweep through depths every 16
                cv2.imshow('[REG] Depth:{}'.format(depth), cost_reg.permute(3,4,2,0,1)[:,:,depth,0,0].cpu().detach().numpy())
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # DEBUG END  
        
        cost_reg = cost_reg.squeeze(1)            # [B, 1, D, H, W]
        prob_volume = F.softmax(cost_reg, dim=1)  # [B, D, H, W]
        
        # DEBUG: plot depths proba
        if '3' in get_powers(self.debug):
            import cv2            
            for depth in range(0,prob_volume.shape[1],16): # sweep through depths every 16
                cv2.imshow('[PROBA] Depth:{}'.format(depth), prob_volume.permute(2,3,1,0)[:,:,depth,0].cpu().detach().numpy())
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # DEBUG END  
        
        depth = depth_regression(prob_volume, depth_values=depth_values)
        
        #  DEBUG: plot depth expectation
        if '4' in get_powers(self.debug): # add 16
            import cv2
            cv2.imshow('[DEPTH EXPECT.]', depth.permute(1,2,0)[:,:,0].cpu().detach().numpy()/depth[0,:,:].cpu().detach().numpy().max())
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # OLI DEBUG END  
        
        with torch.no_grad():
            # photometric confidence: extracts the probability at the depth indices
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)   # [B, D, H, W]
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()    # [B, H, W]
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)                                           # [B, H, W]
        
        #  DEBUG: plot photometric confidence
        if '5' in get_powers(self.debug): # add 32
            import cv2
            proba = photometric_confidence.permute(1,2,0)[:,:,0].cpu().numpy()
            cv2.imshow('[photometric confidence]', proba)
            for conf_pct in [0.1, 0.25, 0.50, 0.75, 0.9]:
                mask = (proba > conf_pct)
                masked_proba = proba.copy()
                masked_proba[~mask] = 0
                cv2.imshow('[photo-conf>{}]'.format(conf_pct), masked_proba) 
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        #  DEBUG END  

        # step 4. depth map refinement
        if not self.refine:
            return {"depth": depth, "photometric_confidence": photometric_confidence}
        else:
            refined_depth = self.refine_network(torch.cat((imgs[0], depth), 1))
            return {"depth": depth, "refined_depth": refined_depth, "photometric_confidence": photometric_confidence}


def mvsnet_loss(depth_est, depth_gt, mask):
    mask = mask > 0.5
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
