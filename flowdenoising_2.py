#!/usr/bin/env python
'''3D Gaussian filtering controlled by the optical flow.
'''

# "flowdenoising.py" is part of "https://github.com/microscopy-processing/FlowDenoising", authored by:
#
# * J.J. Fernández (CSIC).
# * V. González-Ruiz (UAL).
#
# Please, refer to the LICENSE to know the terms of usage of this software.

import logging
import os
import numpy as np
import cv2
import scipy.ndimage
import time
import imageio
import tifffile
import skimage.io
import mrcfile
import argparse
import threading
import time

import concurrent
import multiprocessing
from multiprocessing import shared_memory
from concurrent.futures.process import ProcessPoolExecutor
__number_of_CPUs__ = multiprocessing.cpu_count()

LOGGING_FORMAT = "[%(asctime)s] (%(levelname)s) %(message)s"

__percent__ = 0

def get_gaussian_kernel(sigma=1):
    logging.info(f"Computing gaussian kernel with sigma={sigma}")
    number_of_coeffs = 3
    number_of_zeros = 0
    while number_of_zeros < 2 :
        delta = np.zeros(number_of_coeffs)
        delta[delta.size//2] = 1
        coeffs = scipy.ndimage.gaussian_filter1d(delta, sigma=sigma)
        number_of_zeros = coeffs.size - np.count_nonzero(coeffs)
        number_of_coeffs += 1
    logging.debug("Kernel computed")
    return coeffs[1:-1]

OFCA_EXTENSION_MODE = cv2.BORDER_REPLICATE
OF_LEVELS = 0
OF_WINDOW_SIDE = 5
OF_ITERS = 3
OF_POLY_N = 5
OF_POLY_SIGMA = 1.2
SIGMA = 2.0

def warp_slice(reference, flow):
    height, width = flow.shape[:2]
    map_x = np.tile(np.arange(width), (height, 1))
    map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
    map_xy = (flow + np.dstack((map_x, map_y))).astype('float32')
    warped_slice = cv2.remap(reference, map_xy, None, interpolation=cv2.INTER_LINEAR, borderMode=OFCA_EXTENSION_MODE)
    return warped_slice

def get_flow(reference, target, l=OF_LEVELS, w=OF_WINDOW_SIDE, prev_flow=None):
    if __debug__:
        time_0 = time.process_time()
    flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=prev_flow, pyr_scale=0.5, levels=l, winsize=w, iterations=OF_ITERS, poly_n=OF_POLY_N, poly_sigma=OF_POLY_SIGMA, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    #flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=None, pyr_scale=0.5, levels=l, winsize=w, iterations=OF_ITERS, poly_n=OF_POLY_N, poly_sigma=OF_POLY_SIGMA, flags=0)
    if __debug__:
        time_1 = time.process_time()
        logging.debug(f"OF computed in {1000*(time_1 - time_0):4.3f} ms, max_X={np.max(flow[0]):+3.2f}, min_X={np.min(flow[0]):+3.2f}, max_Y={np.max(flow[1]):+3.2f}, min_Y={np.min(flow[1]):+3.2f}")
    return flow

def OF_filter_along_Z_slice(z, padded_vol, kernel):
    tmp_slice = np.zeros_like(__vol__[z, :, :]).astype(np.float32)
    assert kernel.size % 2 != 0 # kernel.size must be odd
    prev_flow = np.zeros(shape=(__vol__.shape[1], __vol__.shape[2], 2), dtype=np.float32)
    for i in range((kernel.size//2) - 1, -1, -1):
        flow = get_flow(padded_vol[z + i, :, :], __vol__[z, :, :], l, w, prev_flow)
        prev_flow = flow
        OF_compensated_slice = warp_slice(padded_vol[z + i, :, :], flow)
        tmp_slice += OF_compensated_slice * kernel[i]
    tmp_slice += __vol__[z, :, :] * kernel[kernel.size//2]
    prev_flow = np.zeros(shape=(__vol__.shape[1], __vol__.shape[2], 2), dtype=np.float32)
    for i in range(kernel.size//2+1, kernel.size):
        flow = get_flow(padded_vol[z + i, :, :], __vol__[z, :, :], l, w, prev_flow)
        prev_flow = flow
        OF_compensated_slice = warp_slice(padded_vol[z + i, :, :], flow)
        tmp_slice += OF_compensated_slice * kernel[i]
    __vol__[z, :, :] = tmp_slice
    #__percent__ = int(100*(z/Z_dim))

def OF_filter_along_Z_chunk(i, padded_vol, kernel):
    Z_dim = __vol__.shape[0]
    for z in range(Z_dim//__number_of_CPUs__):
        # Notice that the slices of a chunk are not contiguous
        OF_filter_along_Z_slice(z*__number_of_CPUs__ + i,
                                padded_vol,
                                kernel)
    for z in range(Z_dim % __number_of_CPUs__):
        OF_filter_along_Z_slice(z*__number_of_CPUs__ + i,
                                padded_vol,
                                kernel)
    return i

def OF_filter_along_Z(kernel, l, w, mean):
    global __percent__
    logging.info(f"Filtering along Z with l={l}, w={w}, and kernel length={kernel.size}")

    if __debug__:
        time_0 = time.process_time()
        min_OF = 1000
        max_OF = -1000

    shape_of_vol = np.shape(__vol__)
    padded_vol = np.full(shape=(shape_of_vol[0] + kernel.size, shape_of_vol[1], shape_of_vol[2]), fill_value=mean)
    padded_vol[kernel.size//2:shape_of_vol[0] + kernel.size//2, :, :] = __vol__
    #for i in range(__number_of_CPUs__):
    #    OF_filter_along_Z_chunk(i, padded_vol, kernel)
    vol_indexes = [i for i in range(__number_of_CPUs__)]
    padded_vols = [padded_vol]*__number_of_CPUs__
    kernels = [kernel]*__number_of_CPUs__
    with ProcessPoolExecutor(max_workers=__number_of_CPUs__) as executor:
        for _ in executor.map(OF_filter_along_Z_chunk,
                              vol_indexes,
                              padded_vols,
                              kernels):
            print(_)

    if __debug__:
        time_1 = time.process_time()
        logging.debug(f"Filtering along Z spent {time_1 - time_0} seconds")
        logging.debug(f"Min OF val: {min_OF}")
        logging.debug(f"Max OF val: {max_OF}")

def OF_filter_along_Y_slice(y, padded_vol, kernel):
    tmp_slice = np.zeros_like(__vol__[:, y, :]).astype(np.float32)
    assert kernel.size % 2 != 0 # kernel.size must be odd
    prev_flow = np.zeros(shape=(__vol__.shape[0], __vol__.shape[2], 2), dtype=np.float32)
    for i in range((kernel.size//2) - 1, -1, -1):
        flow = get_flow(padded_vol[:, y + i, :], __vol__[:, y, :], l, w, prev_flow)
        prev_flow = flow
        OF_compensated_slice = warp_slice(padded_vol[:, y + i, :], flow)
        tmp_slice += OF_compensated_slice * kernel[i]
    tmp_slice += __vol__[:, y, :] * kernel[kernel.size//2]
    prev_flow = np.zeros(shape=(__vol__.shape[0], __vol__.shape[2], 2), dtype=np.float32)
    for i in range(kernel.size//2+1, kernel.size):
        flow = get_flow(padded_vol[:, y + i, :], __vol__[:, y, :], l, w, prev_flow)
        prev_flow = flow
        OF_compensated_slice = warp_slice(padded_vol[:, y + i, :], flow)
        tmp_slice += OF_compensated_slice * kernel[i]
    __vol__[:, y, :] = tmp_slice
    #__percent__ = int(100*(y/Y_dim))

def OF_filter_along_Y_chunk(i, padded_vol, kernel):
    Y_dim = __vol__.shape[1]
    for y in range(Y_dim//__number_of_CPUs__):
        # Notice that the slices of a chunk are not contiguous
        OF_filter_along_Y_slice(y*__number_of_CPUs__ + i,
                                padded_vol,
                                kernel)
    for y in range(Y_dim % __number_of_CPUs__):
        OF_filter_along_Y_slice(y*__number_of_CPUs__ + i,
                                padded_vol,
                                kernel)
    return i

def OF_filter_along_Y(kernel, l, w, mean):
    global __percent__
    logging.info(f"Filtering along Y with l={l}, w={w}, and kernel length={kernel.size}")
    if __debug__:
        time_0 = time.process_time()
        min_OF = 1000
        max_OF = -1000 
    shape_of_vol = np.shape(__vol__)
    padded_vol = np.full(shape=(shape_of_vol[0], shape_of_vol[1] + kernel.size, shape_of_vol[2]), fill_value=mean)
    padded_vol[:, kernel.size//2:shape_of_vol[1] + kernel.size//2, :] = __vol__
    #for i in range(__number_of_CPUs__):
    #    OF_filter_along_Y_chunk(i, padded_vol, kernel)
    vol_indexes = [i for i in range(__number_of_CPUs__)]
    padded_vols = [padded_vol]*__number_of_CPUs__
    kernels = [kernel]*__number_of_CPUs__
    with ProcessPoolExecutor(max_workers=__number_of_CPUs__) as executor:
        for _ in executor.map(OF_filter_along_Y_chunk,
                              vol_indexes,
                              padded_vols,
                              kernels):
            print(_)

    if __debug__:
        time_1 = time.process_time()
        logging.debug(f"Filtering along Y spent {time_1 - time_0} seconds")
        logging.debug(f"Min OF val: {min_OF}")
        logging.debug(f"Max OF val: {max_OF}")

def OF_filter_along_X_slice(x, padded_vol, kernel):
    tmp_slice = np.zeros_like(__vol__[:, :, x]).astype(np.float32)
    assert kernel.size % 2 != 0 # kernel.size must be odd
    prev_flow = np.zeros(shape=(__vol__.shape[0], __vol__.shape[1], 2), dtype=np.float32)
    for i in range((kernel.size//2) - 1, -1, -1):
        flow = get_flow(padded_vol[:, :, x + i], __vol__[:, :, x], l, w, prev_flow)
        prev_flow = flow
        OF_compensated_slice = warp_slice(padded_vol[:, :, x + i], flow)
        tmp_slice += OF_compensated_slice * kernel[i]
    tmp_slice += __vol__[:, :, x] * kernel[kernel.size//2]
    prev_flow = np.zeros(shape=(__vol__.shape[0], __vol__.shape[1], 2), dtype=np.float32)
    for i in range(kernel.size//2+1, kernel.size):
        flow = get_flow(padded_vol[:, :, x + i], __vol__[:, :, x], l, w, prev_flow)
        prev_flow = flow
        OF_compensated_slice = warp_slice(padded_vol[:, :, x + i], flow)
        tmp_slice += OF_compensated_slice * kernel[i]
    __vol__[:, :, x] = tmp_slice
    #__percent__ = int(100*(x/X_dim))

def OF_filter_along_X_chunk(i, padded_vol, kernel):
    X_dim = __vol__.shape[2]
    for x in range(X_dim//__number_of_CPUs__):
        # Notice that the slices of a chunk are not contiguous
        OF_filter_along_X_slice(x*__number_of_CPUs__ + i,
                                padded_vol,
                                kernel)
    for x in range(X_dim % __number_of_CPUs__):
        OF_filter_along_X_slice(x*__number_of_CPUs__ + i,
                                padded_vol,
                                kernel)
    return i
    
def OF_filter_along_X(kernel, l, w, mean):
    global __percent__
    logging.info(f"Filtering along X with l={l}, w={w}, and kernel length={kernel.size}")
    if __debug__:
        time_0 = time.process_time()
        min_OF = 1000
        max_OF = -1000
    shape_of_vol = np.shape(__vol__)
    padded_vol = np.full(shape=(shape_of_vol[0], shape_of_vol[1], shape_of_vol[2] + kernel.size), fill_value=mean)
    padded_vol[:, :, kernel.size//2:shape_of_vol[2] + kernel.size//2] = __vol__
    #for i in range(__number_of_CPUs__):
    #    OF_filter_along_X_chunk(i, padded_vol, kernel)
    vol_indexes = [i for i in range(__number_of_CPUs__)]
    padded_vols = [padded_vol]*__number_of_CPUs__
    kernels = [kernel]*__number_of_CPUs__
    with ProcessPoolExecutor(max_workers=__number_of_CPUs__) as executor:
        for _ in executor.map(OF_filter_along_X_chunk,
                              vol_indexes,
                              padded_vols,
                              kernels):
            print(_)

    if __debug__:
        time_1 = time.process_time()
        logging.debug(f"Filtering along X spent {time_1 - time_0} seconds")

def OF_filter(kernels, l, w):
    mean = __vol__.mean()
    OF_filter_along_Z(kernels[0], l, w, mean)
    print("__vol__.mean after Z =", __vol__.mean())
    OF_filter_along_Y(kernels[1], l, w, mean)
    print("__vol__.mean() after Y =", __vol__.mean())
    OF_filter_along_X(kernels[2], l, w, mean)

###############################################################

def no_OF_filter_along_Z_slice(z, padded_vol, kernel):
    tmp_slice = np.zeros(shape=(__vol__.shape[1], __vol__.shape[2]),
                         dtype=np.float32)
    for i in range(kernel.size):
        tmp_slice += padded_vol[z + i, :, :] * kernel[i]
    __vol__[z, :, :] = tmp_slice
    #__percent__ = int(100*(z/Z_dim))

def no_OF_filter_along_Z_chunk(i, padded_vol, kernel):
    Z_dim = __vol__.shape[0]
    for z in range(Z_dim//__number_of_CPUs__):
        # Notice that the slices of a chunk are not contiguous
        no_OF_filter_along_Z_slice(z*__number_of_CPUs__ + i,
                                   padded_vol,
                                   kernel)
    for z in range(Z_dim % __number_of_CPUs__):
        no_OF_filter_along_Z_slice(z*__number_of_CPUs__ + i,
                                   padded_vol,
                                   kernel)
    return i

def no_OF_filter_along_Z(kernel, mean):
    global __percent__
    logging.info(f"Filtering along Z with kernel length={kernel.size}")

    if __debug__:
        time_0 = time.process_time()

    padded_vol = np.full(shape=(__vol__.shape[0] + kernel.size,
                                __vol__.shape[1],
                                __vol__.shape[2]),
                         fill_value=mean)
    padded_vol[kernel.size//2:__vol__.shape[0] + kernel.size//2, ...] = __vol__

    #for i in range(__number_of_CPUs__):
    #    no_OF_filter_along_Z_chunk(i, padded_vol, kernel)
    vol_indexes = [i for i in range(__number_of_CPUs__)]
    padded_vols = [padded_vol]*__number_of_CPUs__
    kernels = [kernel]*__number_of_CPUs__
    with ProcessPoolExecutor(max_workers=__number_of_CPUs__) as executor:
        for _ in executor.map(no_OF_filter_along_Z_chunk,
                              vol_indexes,
                              padded_vols,
                              kernels):
            print(_)

    if __debug__:
        time_1 = time.process_time()
        logging.debug(f"Filtering along Z spent {time_1 - time_0} seconds")

def no_OF_filter_along_Y_slice(y, padded_vol, kernel):
    tmp_slice = np.zeros(shape=(__vol__.shape[0], __vol__.shape[2]),
                         dtype=np.float32)
    for i in range(kernel.size):
        tmp_slice += padded_vol[:, y + i, :] * kernel[i]
    __vol__[:, y, :] = tmp_slice
    #__percent__ = int(100*(y/Y_dim))
        
def no_OF_filter_along_Y_chunk(i, padded_vol, kernel):
    Y_dim = __vol__.shape[1]
    for y in range(Y_dim//__number_of_CPUs__):
        # Notice that the slices of a chunk are not contiguous
        no_OF_filter_along_Y_slice(y*__number_of_CPUs__ + i,
                                   padded_vol,
                                   kernel)
    for y in range(Y_dim % __number_of_CPUs__):
        no_OF_filter_along_Y_slice(y*__number_of_CPUs__ + i,
                                   padded_vol,
                                   kernel)
    return i

def no_OF_filter_along_Y(kernel, mean):
    global __percent__
    logging.info(f"Filtering along Y with kernel length={kernel.size}")

    if __debug__:
        time_0 = time.process_time()

    padded_vol = np.full(shape=(__vol__.shape[0],
                                __vol__.shape[1] + kernel.size,
                                __vol__.shape[2]),
                         fill_value=mean)
    padded_vol[:, kernel.size//2:__vol__.shape[1] + kernel.size//2, :] = __vol__

    #for i in range(__number_of_CPUs__):
    #    no_OF_filter_along_Y_chunk(i, __vol__, padded_vol, kernel)
    vol_indexes = [i for i in range(__number_of_CPUs__)]
    padded_vols = [padded_vol]*__number_of_CPUs__
    kernels = [kernel]*__number_of_CPUs__
    with ProcessPoolExecutor(max_workers=__number_of_CPUs__) as executor:
        for _ in executor.map(no_OF_filter_along_Y_chunk,
                              vol_indexes,
                              padded_vols,
                              kernels):
            print(_)

    if __debug__:
        time_1 = time.process_time()
        logging.debug(f"Filtering along Y spent {time_1 - time_0} seconds")

def no_OF_filter_along_X_slice(x, padded_vol, kernel):
    tmp_slice = np.zeros(shape=(__vol__.shape[0], __vol__.shape[1]),
                         dtype=np.float32)
    for i in range(kernel.size):
        tmp_slice += padded_vol[:, :, x + i]*kernel[i]
    __vol__[:, :, x] = tmp_slice
    #__percent__ = int(100*(x/X_dim))

def no_OF_filter_along_X_chunk(i, padded_vol, kernel):
    X_dim = __vol__.shape[2]
    for x in range(X_dim//__number_of_CPUs__):
        # Notice that the slices of a chunk are not contiguous
        no_OF_filter_along_X_slice(x*__number_of_CPUs__ + i,
                                   padded_vol,
                                   kernel)
    for x in range(X_dim % __number_of_CPUs__):
        no_OF_filter_along_X_slice(x*__number_of_CPUs__ + i,
                                   padded_vol,
                                   kernel)
    return i

def no_OF_filter_along_X(kernel, mean):
    global __percent__
    logging.info(f"Filtering along X with kernel length={kernel.size}")
    if __debug__:
        time_0 = time.process_time()
    padded_vol = np.full(shape=(__vol__.shape[0],
                                __vol__.shape[1],
                                __vol__.shape[2] + kernel.size),
                         fill_value=mean)
    padded_vol[:, :, kernel.size//2:__vol__.shape[2] + kernel.size//2] = __vol__

    #for i in range(__number_of_CPUs__):
    #    no_OF_filter_along_X_chunk(i, __vol__, padded_vol, kernel)
    vol_indexes = [i for i in range(__number_of_CPUs__)]
    padded_vols = [padded_vol]*__number_of_CPUs__
    kernels = [kernel]*__number_of_CPUs__
    with ProcessPoolExecutor(max_workers=__number_of_CPUs__) as executor:
        for _ in executor.map(no_OF_filter_along_X_chunk,
                              vol_indexes,
                              padded_vols,
                              kernels):
            print(_)
    if __debug__:
        time_1 = time.process_time()
        logging.debug(f"Filtering along X spent {time_1 - time_0} seconds")

def no_OF_filter(kernels):
    mean = __vol__.mean()
    no_OF_filter_along_Z(kernels[0], mean)
    no_OF_filter_along_Y(kernels[1], mean)
    no_OF_filter_along_X(kernels[2], mean)
    #no_OF_filter_along_X(__vol__[: ,: ,:__vol__.shape[2]//2 + kernel[2].size//2], kernel[2], mean)
    #no_OF_filter_along_X(__vol__[: ,: ,__vol__.shape[2]//2 - kernel[2].size//2:], kernel[2], mean)
    
def int_or_str(text):
    '''Helper function for argument parsing.'''
    try:
        return int(text)
    except ValueError:
        return text

def feedback():
    global __percent__
    while True:
        logging.info(f"{__percent__} %")
        time.sleep(1)

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument("-t", "--transpose", nargs="+",
#                    help="Transpose pattern (see https://numpy.org/doc/stable/reference/generated/numpy.transpose.html, by default the 3D volume in not transposed)",
#                    default=(0, 1, 2))
parser.add_argument("-i", "--input", type=int_or_str,
                    help="Input a MRC-file or a multi-image TIFF-file",
                    default="./volume.mrc")
parser.add_argument("-o", "--output", type=int_or_str,
                    help="Output a MRC-file or a multi-image TIFF-file",
                    default="./denoised_volume.mrc")
#parser.add_argument("-n", "--number_of_images", type=int_or_str,
#                    help="Number of input images (only if the sequence of images is input)",
#                    default=32)
parser.add_argument("-s", "--sigma", nargs="+",
                    help="Gaussian sigma for each dimension in the order (Z, Y, X)",
                    default=(SIGMA, SIGMA, SIGMA))
                    #default=f"{SIGMA} {SIGMA} {SIGMA}")
parser.add_argument("-l", "--levels", type=int_or_str,
                    help="Number of levels of the Gaussian pyramid used by the optical flow estimator",
                    default=OF_LEVELS)
parser.add_argument("-w", "--winside", type=int_or_str,
                    help="Side of the window used by the optical flow estimator",
                    default=OF_WINDOW_SIDE)
parser.add_argument("-v", "--verbosity", type=int_or_str,
                    help="Verbosity level", default=0)
parser.add_argument("-n", "--no_OF", action="store_true", help="Disable optical flow compensation")
parser.add_argument("-m", "--memory_map", action="store_true", help="Enable memory-mapping (see https://mrcfile.readthedocs.io/en/stable/usage_guide.html#dealing-with-large-files, only for MRC files)")

def show_memory_usage(msg=''):
    logging.info(f"{psutil.Process(os.getpid()).memory_info().rss/(1024*1024):.1f} MB used in process {os.getpid()} {msg}")

if __name__ == "__main__":

    parser.description = __doc__
    args = parser.parse_args()
    
    if args.verbosity == 2:
        logging.basicConfig(format=LOGGING_FORMAT, level=logging.DEBUG)
        logging.info("Verbosity level = 2")
    elif args.verbosity == 1:
        logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)
        logging.info("Verbosity level = 1")        
    else:
        logging.basicConfig(format=LOGGING_FORMAT, level=logging.CRITICAL)

    thread = threading.Thread(target=feedback)
    thread.daemon = True # To obey CTRL+C interruption.
    thread.start()
        
    sigma = [float(i) for i in args.sigma]
    logging.info(f"sigma={tuple(sigma)}")
    l = args.levels
    w = args.winside
    
    #logging.debug(f"Using transpose pattern {args.transpose} {type(args.transpose)}")
    #transpose_pattern = tuple([int(i) for i in args.transpose])
    #logging.info(f"transpose={transpose_pattern}")

    if __debug__:
        logging.info(f"reading \"{args.input}\"")
        time_0 = time.process_time()

    logging.debug(f"input = {args.input}")

    MRC_input = ( args.input.split('.')[-1] == "MRC" or args.input.split('.')[-1] == "mrc" )
    if MRC_input:
        if args.memory_map:
            logging.info(f"Using memory mapping")
            vol_MRC = rc = mrcfile.mmap(args.input, mode='r+')
        else:
            vol_MRC = mrcfile.open(args.input, mode="r+")
        __vol__ = vol_MRC.data
    else:
        __vol__ = skimage.io.imread(args.input, plugin="tifffile").astype(np.float32)
    size = __vol__.dtype.itemsize * __vol__.size
    logging.info(f"vol requires {size/(1024*1024):.1f} MB")

    # Copy to shared memory
    SM_padded_vol = shared_memory.SharedMemory(
        create=True,
        size=(__vol__.shape[0] + kernel[0].size)*(__vol__.shape[1] + kernel[1])*(__vol__.shape[2] + kernel[2].size)*__vol__.dtype.itemsize,
        name="padded_vol") # See /dev/shm/
    np_vol = np.ndarray(
        shape=__vol__.shape,
        dtype=__vol__.dtype,
        buffer=SM_vol.buf)
    np_vol[...] = __vol__[...]
    __vol__ = np_vol

    logging.info(f"shape of the input volume (Z, Y, X) = {__vol__.shape}")
    logging.info(f"type of the volume = {__vol__.dtype}")

    #vol = np.transpose(vol, transpose_pattern)
    #logging.info(f"shape of the volume to denoise (Z, Y, X) = {vol.shape}")

    if __debug__:
        time_1 = time.process_time()
        logging.info(f"read \"{args.input}\" in {time_1 - time_0} seconds")

    logging.info(f"{args.input} type = {__vol__.dtype}")
    logging.info(f"{args.input} max = {__vol__.max()}")
    logging.info(f"{args.input} min = {__vol__.min()}")
    vol_mean = __vol__.mean()
    logging.info(f"Input vol average = {vol_mean}")

    kernels = [None]*3
    kernels[0] = get_gaussian_kernel(sigma[0])
    kernels[1] = get_gaussian_kernel(sigma[1])
    kernels[2] = get_gaussian_kernel(sigma[2])
    logging.info(f"length of each filter (Z, Y, X) = {[len(i) for i in [*kernels]]}")

    if args.no_OF:
        no_OF_filter(kernels)
    else:
         OF_filter(kernels, l, w)

    #filtered_vol = np.transpose(filtered_vol, transpose_pattern)
    logging.info(f"shape of the denoised volume (Z, Y, X) = {__vol__.shape}")

    logging.info(f"{args.output} type = {__vol__.dtype}")
    logging.info(f"{args.output} max = {__vol__.max()}")
    logging.info(f"{args.output} min = {__vol__.min()}")
    logging.info(f"Output vol average = {__vol__.mean()}")
    
    if __debug__:
        logging.info(f"writting \"{args.output}\"")
        time_0 = time.process_time()

    logging.debug(f"output = {args.output}")

    MRC_output = ( args.output.split('.')[-1] == "MRC" or args.output.split('.')[-1] == "mrc" )

    if MRC_output:
        logging.debug(f"Writting MRC file")
        with mrcfile.new(args.output, overwrite=True) as mrc:
            mrc.set_data(__vol__.astype(np.float32))
            mrc.data
    else:
        if np.max(__vol__) < 256:
            logging.debug(f"Writting TIFF file (uint8)")
            skimage.io.imsave(args.output, __vol__.astype(np.uint8), plugin="tifffile")
        else:
            logging.debug(f"Writting TIFF file (uint16)")
            skimage.io.imsave(args.output, __vol__.astype(np.uint16), plugin="tifffile")

    SM_vol.close()
    SM_vol.unlink()
    
    if __debug__:
        time_1 = time.process_time()        
        logging.info(f"written \"{args.output}\" in {time_1 - time_0} seconds")
