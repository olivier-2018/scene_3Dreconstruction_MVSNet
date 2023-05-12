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
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import gc
import sys
import datetime
import random

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='A PyTorch Implementation of MVSNet')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--refine', action='store_true', help='use the refine network')

parser.add_argument('--dataset', default='dtu_yao', choices=['dtu_yao', 'blender'],help='select dataset')
parser.add_argument('--trainpath', default="", help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', default="lists/dtu/train.txt", help='train list')
parser.add_argument('--testlist', default="lists/dtu/test.txt", help='test list')
parser.add_argument('--pairfile', default="pair.txt", help='pair filename')

parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values (DTU=1.06)')
parser.add_argument('--Nlights', type=int, default=7, help='number of light sources in the dataset (DTU=7)')
parser.add_argument('--NtrainViews', type=int, default=3, help='number of views used for training (DTU=3)')
parser.add_argument('--NtestViews', type=int, default=5, help='number of views used for training (DTU=5)')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./outputs/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=100, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='0 for random seed')

parser.add_argument('--debug_MVSnet', type=int, default=0, help='powers of 2 for switches selection (debug = 2⁰+2¹+2³+2⁴+...) with '
                    '0: print matrices and plot features (add 1) '
                    '1: plot warped views (add 2) '
                    '2: plot regularization (add 4) '
                    '3: plot depths proba (add 8) '
                    '4: plot expectation (add 16) '
                    '5: plot photometric confidence (add 32) '
                    '63: ALL')


args = parser.parse_args()

# multi-debug function
####################### 
def get_powers(n):
    return [str(p) for p,v in enumerate(bin(n)[:1:-1]) if int(v)]


# check things if resume
################################################
if args.resume:
    assert args.mode == "train", 'Resume run requested but training not requested (set --mode to train)'
    assert args.loadckpt is None, 'Resume run requested but specific loadpoint also requested (unset --loadckpt)'
if args.testpath is None:
    args.testpath = args.trainpath

# setting seed default
###########################
torch.cuda.empty_cache() # OLI
if args.seed == 0:
    seed = random.randint(1,99999999)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
else:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed) 


# create logger for mode "train" and "testall"
##################################################
if args.mode == "train":
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)

# dataset, dataloader
##################################################
print ("Building dataloaders.")
MVSDataset = find_dataset_def(args.dataset)
train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", args.NtrainViews, args.numdepth, args.interval_scale, Nlights=args.Nlights, pairfile=args.pairfile)
test_dataset = MVSDataset(args.testpath, args.testlist, "test", args.NtestViews, args.numdepth, args.interval_scale, Nlights=args.Nlights, pairfile=args.pairfile)

TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=10, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=10, drop_last=False)

# model, optimizer, loss
##################################################
print ("Initializing model and sending to cuda.")
model = MVSNet(refine=args.refine, debug=args.debug_MVSnet)
if args.mode in ["train", "test"]:
    model = nn.DataParallel(model)
model.cuda()
model_loss = mvsnet_loss
print ("Initializing optimizer.")
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

# load parameters
##################################################
start_epoch = 0
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("Resuming from ", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("Loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])

print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


##################################################
# main functions
##################################################
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma, last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        lr_scheduler.step()
        global_step = len(TrainImgLoader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
                logger.flush()
                print('Epoch {}/{}, Iter {}/{}, LR:{:.2E}, loss={:.3f}, abs_depth_err={:.3f}, thres1mm_error={:.3f}, thres2mm_error={:.3f}, thres4mm_error={:.3f}, thres8mm_error={:.3f}, time={:.3f}'.format(
                        epoch_idx, args.epochs, 
                        batch_idx,len(TrainImgLoader), 
                        optimizer.param_groups[0]["lr"],
                        loss,
                        scalar_outputs["abs_depth_error"],
                        scalar_outputs["thres1mm_error"],
                        scalar_outputs["thres2mm_error"],
                        scalar_outputs["thres4mm_error"],
                        scalar_outputs["thres8mm_error"],
                        time.time() - start_time))
                sys.stdout.flush()
            del scalar_outputs, image_outputs

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))

        # testing
        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
                print('Epoch {}/{}, Iter {}/{}, LR:{:.2E}, loss={:.3f}, abs_depth_err={:.3f}, thres1mm_error={:.3f}, thres2mm_error={:.3f}, thres4mm_error={:.3f}, thres8mm_error={:.3f}, time={:.3f}'.format(
                        epoch_idx, args.epochs, 
                        batch_idx,len(TestImgLoader), 
                        optimizer.param_groups[0]["lr"],
                        loss,
                        scalar_outputs["abs_depth_error"],
                        scalar_outputs["thres1mm_error"],
                        scalar_outputs["thres2mm_error"],
                        scalar_outputs["thres4mm_error"],
                        scalar_outputs["thres8mm_error"],
                        time.time() - start_time))
                sys.stdout.flush()
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
        save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())
        # gc.collect()


def test():
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=True)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss, time.time() - start_time))
        if batch_idx % 100 == 0:
            print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    print("final", avg_test_scalars)


def train_sample(sample, detailed_summary=False):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"] 

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]
    loss = model_loss(depth_est, depth_gt, mask)
    
    photo_conf = outputs['photometric_confidence']
    mask_conf_50pct = (photo_conf > 0.5)
    errormap = (depth_est - depth_gt).abs() * mask

    scalar_outputs = {"loss": loss}
    scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
    scalar_outputs["thres1mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 1)
    scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
    scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
    scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)
    
    image_outputs = {"errormap": errormap,
                     "photo_conf": photo_conf, 
                     "depth_est": depth_est * mask, 
                     "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     }
    if detailed_summary:
        
        mask_errormap_1mm = (errormap < 1.0)
        masked_em1 = errormap.cpu().detach().numpy().copy()
        masked_em1[~mask_errormap_1mm.cpu().detach().numpy()] = 0.0
        masked_em1[mask_errormap_1mm.cpu().detach().numpy()] = 1.0
        masked_em1[~(mask.cpu().detach().numpy()>0.5)] = 0.0
        image_outputs["errormap_1mm_mask"] = masked_em1
        
        mask_errormap_2mm = (errormap < 2.0)
        masked_em2 = errormap.cpu().detach().numpy().copy()
        masked_em2[~mask_errormap_2mm.cpu().detach().numpy()] = 0.0
        masked_em2[mask_errormap_2mm.cpu().detach().numpy()] = 1.0
        masked_em2[~(mask.cpu().detach().numpy()>0.5)] = 0.0
        image_outputs["errormap_2mm_mask"] = masked_em2
                
        masked_conf = photo_conf.cpu().detach().numpy().copy()
        masked_conf[~mask_conf_50pct.cpu().detach().numpy()] = 0.0
        masked_conf[mask_conf_50pct.cpu().detach().numpy()] = 1.0
        masked_conf[~(mask.cpu().detach().numpy()>0.5)] = 0.0
        image_outputs["photo_conf_50pct"] = masked_conf
        
        image_outputs["mask"] = sample["mask"]


    loss.backward()
    optimizer.step()
    
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


@make_nograd_func
def test_sample(sample, detailed_summary=True):
    model.eval()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]
    loss = model_loss(depth_est, depth_gt, mask)
   
    photo_conf = outputs['photometric_confidence']
    mask_conf_50pct = (photo_conf > 0.5)
    errormap = (depth_est - depth_gt).abs() * mask


    scalar_outputs = {"loss": loss}
    scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
    scalar_outputs["thres1mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 1)
    scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
    scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
    scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)
    
    image_outputs = {"errormap": errormap,
                    "photo_conf": photo_conf, 
                    "depth_est": depth_est * mask, 
                    "depth_gt": sample["depth"],
                    "ref_img": sample["imgs"][:, 0],
                }
    
    if detailed_summary:
        
        mask_errormap_1mm = (errormap < 1.0)
        masked_em1 = errormap.cpu().detach().numpy().copy()
        masked_em1[~mask_errormap_1mm.cpu().detach().numpy()] = 0.0
        masked_em1[mask_errormap_1mm.cpu().detach().numpy()] = 1.0
        masked_em1[~(mask.cpu().detach().numpy()>0.5)] = 0.0
        image_outputs["errormap_1mm_mask"] = masked_em1
        
        mask_errormap_2mm = (errormap < 2.0)
        masked_em2 = errormap.cpu().detach().numpy().copy()
        masked_em2[~mask_errormap_2mm.cpu().detach().numpy()] = 0.0
        masked_em2[mask_errormap_2mm.cpu().detach().numpy()] = 1.0
        masked_em2[~(mask.cpu().detach().numpy()>0.5)] = 0.0
        image_outputs["errormap_2mm_mask"] = masked_em2
                
        masked_conf = photo_conf.cpu().detach().numpy().copy()
        masked_conf[~mask_conf_50pct.cpu().detach().numpy()] = 0.0
        masked_conf[mask_conf_50pct.cpu().detach().numpy()] = 1.0
        masked_conf[~(mask.cpu().detach().numpy()>0.5)] = 0.0
        image_outputs["photo_conf_50pct"] = masked_conf
        
        image_outputs["mask"] = sample["mask"]
        
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


def profile():
    warmup_iter = 5
    iter_dataloader = iter(TestImgLoader)

    @make_nograd_func
    def do_iteration():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        test_sample(next(iter_dataloader), detailed_summary=True)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    for i in range(warmup_iter):
        t = do_iteration()
        print('WarpUp Iter {}, time = {:.4f}'.format(i, t))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for i in range(5):
            t = do_iteration()
            print('Profile Iter {}, time = {:.4f}'.format(i, t))
            time.sleep(0.02)

    if prof is not None:
        # print(prof)
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    elif args.mode == "profile":
        profile()
