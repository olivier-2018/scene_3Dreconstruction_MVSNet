import numpy as np
import re
import sys
import math
from  PIL import Image


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


def read_rescale_crop_img(img_fname, intrinsics, img_res=(512,640), DEBUG=False):
    
    base_image_size = 32
    
    # Read image
    img = Image.open(img_fname)
    w_src, h_src = img.size  # Warning: width first with pillow
    if DEBUG: print(f"[read_rescale_crop_img] img read dims: ({h_src},{w_src})")        
    
    # evaluate scale factor 
    h_target, w_target = img_res
    
    if DEBUG: print(f"[read_rescale_crop_img] img target dims: ({h_target},{w_target})")        
    h_scale = float(h_target) / h_src
    w_scale = float(w_target) / w_src
    if h_scale > 1 or w_scale > 1:
        print("[read_rescale_crop_img] max_h, max_w should < W and H (image resolution should only be reduced)!")
        exit()
    resize_scale = h_scale
    if w_scale > h_scale:
        resize_scale = w_scale
    if DEBUG: print(f"[read_rescale_crop_img] resize_scale: {resize_scale}")        
    
    # rescale image
    img_rescaled = img.resize(size=(int(w_src*resize_scale), int(h_src*resize_scale)), resample=Image.BILINEAR ) # Warning: width first with pillow
    w_rescaled, h_rescaled = img_rescaled.size        
    if DEBUG: print(f"[read_rescale_crop_img] img rescaled dims: ({h_rescaled}, {w_rescaled})")
    
    # rescale intrinsics
    if DEBUG: print("[read_rescale_crop_img] intrinsics:\n", intrinsics)
    intrinsics[:2,:] *= resize_scale
    if DEBUG: print("[read_rescale_crop_img] rescaled intrinsics:\n", intrinsics)
    
    # determine if cropping needed (dims must be compatible with base_image_size)
    final_h = h_rescaled
    final_w = w_rescaled
    
    if final_h > h_target:
        final_h = h_target
    else:
        final_h = int(math.floor(h_target / base_image_size) * base_image_size)
        
    if final_w > w_target:
        final_w = w_target
    else:
        final_w = int(math.floor(w_target / base_image_size) * base_image_size)

    # evaluate cropping parameters 
    start_h = int(math.floor((h_rescaled - final_h) / 2))
    start_w = int(math.floor((w_rescaled - final_w) / 2))
    finish_h = start_h + final_h
    finish_w = start_w + final_w
    
    # crop img and intrinsics
    # img_cropped = img_rescaled[start_h:finish_h, start_w:finish_w] # for numpy
    croping_dims = (start_w, start_h, finish_w, finish_h)
    img_cropped = img_rescaled.crop(croping_dims)  # for pillow
    if DEBUG: 
        print(f"[read_rescale_crop_img] croping dims: (left, top, right, bottom)={croping_dims}")
        print(f"[read_rescale_crop_img] cropped img dims: ({img_cropped.size[1]}, {img_cropped.size[0]})")
    
    # crop intrinsics
    intrinsics[0,-1] -= start_w
    intrinsics[1,-1] -= start_h
    if DEBUG: print("[read_rescale_crop_img] cropped intrinsics:\n", intrinsics)
    
    # convert pillow image to numpy
    np_img = np.array(img_cropped, dtype=np.float32) / 255.
    
    # checks shape
    # assert np_img.shape[:2] == img_res
    
    # check image has 3 channels (RGB), stack if only 1 channel
    if len(np_img.shape) == 2:
        np_img = np.dstack((np_img, np_img, np_img))
        
    if DEBUG: print(f"[read_rescale_crop_img] np_img shape: {np_img.shape}")

    return np_img, intrinsics
