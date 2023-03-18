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
OF_WINDOW_SIZE = 5
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

def get_flow(reference, target, l=OF_LEVELS, w=OF_WINDOW_SIZE, prev_flow=None):
    if __debug__:
        time_0 = time.perf_counter()
    flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=prev_flow, pyr_scale=0.5, levels=l, winsize=w, iterations=OF_ITERS, poly_n=OF_POLY_N, poly_sigma=OF_POLY_SIGMA, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    #flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=None, pyr_scale=0.5, levels=l, winsize=w, iterations=OF_ITERS, poly_n=OF_POLY_N, poly_sigma=OF_POLY_SIGMA, flags=0)
    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"OF computed in {1000*(time_1 - time_0):4.3f} ms, max_X={np.max(flow[0]):+3.2f}, min_X={np.min(flow[0]):+3.2f}, max_Y={np.max(flow[1]):+3.2f}, min_Y={np.min(flow[1]):+3.2f}")
    return flow

def get_flow_(reference, target, l=OF_LEVELS, w=OF_WINDOW_SIZE, prev_flow=None):
    if __debug__:
        time_0 = time.perf_counter()
    flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=prev_flow, pyr_scale=0.5, levels=l, winsize=w, iterations=OF_ITERS, poly_n=OF_POLY_N, poly_sigma=OF_POLY_SIGMA, flags=0)
    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"OF computed in {1000*(time_1 - time_0):4.3f} ms, max_X={np.max(flow[0]):+3.2f}, min_X={np.min(flow[0]):+3.2f}, max_Y={np.max(flow[1]):+3.2f}, min_Y={np.min(flow[1]):+3.2f}")
    return flow

def OF_filter_along_Z(vol, kernel, l, w, mean):
    global __percent__
    logging.info(f"Filtering along Z with l={l}, w={w}, and kernel length={kernel.size}")
    if __debug__:
        time_0 = time.perf_counter()
        min_OF = 1000
        max_OF = -1000 
    filtered_vol = np.zeros_like(vol).astype(np.float32)
    shape_of_vol = np.shape(vol)
    #padded_vol = np.zeros(shape=(shape_of_vol[0] + kernel.size, shape_of_vol[1], shape_of_vol[2]))
    padded_vol = np.full(shape=(shape_of_vol[0] + kernel.size, shape_of_vol[1], shape_of_vol[2]), fill_value=mean)
    padded_vol[kernel.size//2:shape_of_vol[0] + kernel.size//2, :, :] = vol
    Z_dim = vol.shape[0]
    for z in range(Z_dim):
        tmp_slice = np.zeros_like(vol[z]).astype(np.float32)
        assert kernel.size % 2 != 0 # kernel.size must be odd
        prev_flow = np.zeros(shape=(shape_of_vol[1], shape_of_vol[2], 2), dtype=np.float32)
        for i in range((kernel.size//2) - 1, -1, -1):
            #print(i)
            flow = get_flow(padded_vol[z + i, :, :], vol[z, :, :], l, w, prev_flow)
            prev_flow = flow
            if __debug__:
                min_OF_iter = np.min(flow)
                if min_OF_iter < min_OF:
                    min_OF = min_OF_iter
                max_OF_iter = np.max(flow)
                if max_OF < max_OF_iter:
                    max_OF = max_OF_iter
            OF_compensated_slice = warp_slice(padded_vol[z + i, :, :], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        tmp_slice += vol[z, :, :] * kernel[kernel.size//2]
        prev_flow = np.zeros(shape=(shape_of_vol[1], shape_of_vol[2], 2), dtype=np.float32)
        for i in range(kernel.size//2+1, kernel.size):
            #print(i)
            flow = get_flow(padded_vol[z + i, :, :], vol[z, :, :], l, w, prev_flow)
            prev_flow = flow
            if __debug__:
                min_OF_iter = np.min(flow)
                if min_OF_iter < min_OF:
                    min_OF = min_OF_iter
                max_OF_iter = np.max(flow)
                if max_OF < max_OF_iter:
                    max_OF = max_OF_iter
            OF_compensated_slice = warp_slice(padded_vol[z + i, :, :], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        filtered_vol[z, :, :] = tmp_slice
        __percent__ = int(100*(z/Z_dim))
    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"Filtering along Z spent {time_1 - time_0} seconds")
        logging.debug(f"Min OF val: {min_OF}")
        logging.debug(f"Max OF val: {max_OF}")
    return filtered_vol

def OF_filter_along_Z_(vol, kernel, l, w, mean):
    global __percent__
    logging.info(f"Filtering along Z with l={l}, w={w}, and kernel length={kernel.size}")
    if __debug__:
        time_0 = time.perf_counter()
        min_OF = 1000
        max_OF = -1000 
    filtered_vol = np.zeros_like(vol).astype(np.float32)
    shape_of_vol = np.shape(vol)
    #padded_vol = np.zeros(shape=(shape_of_vol[0] + kernel.size, shape_of_vol[1], shape_of_vol[2]))
    padded_vol = np.full(shape=(shape_of_vol[0] + kernel.size, shape_of_vol[1], shape_of_vol[2]), fill_value=mean)
    padded_vol[kernel.size//2:shape_of_vol[0] + kernel.size//2, :, :] = vol
    Z_dim = vol.shape[0]
    prev_flow = None
    for z in range(Z_dim):
        tmp_slice = np.zeros_like(vol[z]).astype(np.float32)
        for i in range(kernel.size):
            if i != kernel.size//2:
                flow = get_flow_(padded_vol[z + i], vol[z], l, w)
                if __debug__:
                    min_OF_iter = np.min(flow)
                    if min_OF_iter < min_OF:
                        min_OF = min_OF_iter
                    max_OF_iter = np.max(flow)
                    if max_OF < max_OF_iter:
                        max_OF = max_OF_iter                        
                OF_compensated_slice = warp_slice(padded_vol[z + i], flow)
                tmp_slice += OF_compensated_slice * kernel[i]
            else:
                tmp_slice += vol[z, :, :] * kernel[i]
        filtered_vol[z, :, :] = tmp_slice
        __percent__ = int(100*(z/Z_dim))
    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"Filtering along Z spent {time_1 - time_0} seconds")
        logging.debug(f"Min OF val: {min_OF}")
        logging.debug(f"Max OF val: {max_OF}")
    return filtered_vol

def no_OF_filter_along_Z(vol, kernel, mean):
    global __percent__
    logging.info(f"Filtering along Z with l={l}, w={w}, and kernel length={kernel.size}")
    if __debug__:
        time_0 = time.perf_counter()
    filtered_vol = np.zeros_like(vol).astype(np.float32)
    shape_of_vol = np.shape(vol)
    #padded_vol = np.zeros(shape=(shape_of_vol[0] + kernel.size, shape_of_vol[1], shape_of_vol[2]))
    padded_vol = np.full(shape=(shape_of_vol[0] + kernel.size, shape_of_vol[1], shape_of_vol[2]), fill_value=mean)
    padded_vol[kernel.size//2:shape_of_vol[0] + kernel.size//2, ...] = vol
    Z_dim = vol.shape[0]
    for z in range(Z_dim):
        tmp_slice = np.zeros_like(vol[z, :, :]).astype(np.float32)
        for i in range(kernel.size):
            tmp_slice += padded_vol[z + i, :, :] * kernel[i]
        filtered_vol[z, :, :] = tmp_slice
        #logging.info(f"Filtering along Z {int(100*(z/Z_dim))}%")
        __percent__ = int(100*(z/Z_dim))
    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"Filtering along Z spent {time_1 - time_0} seconds")
    return filtered_vol

def OF_filter_along_Y_(vol, kernel, l, w, mean):
    global __percent__
    logging.info(f"Filtering along Y with l={l}, w={w}, and kernel length={kernel.size}")
    if __debug__:
        time_0 = time.perf_counter()
        min_OF = 1000
        max_OF = -1000 
    filtered_vol = np.zeros_like(vol).astype(np.float32)
    shape_of_vol = np.shape(vol)
    #padded_vol = np.zeros(shape=(shape_of_vol[0], shape_of_vol[1] + kernel.size, shape_of_vol[2]))
    padded_vol = np.full(shape=(shape_of_vol[0], shape_of_vol[1] + kernel.size, shape_of_vol[2]), fill_value=mean)
    padded_vol[:, kernel.size//2:shape_of_vol[1] + kernel.size//2, :] = vol
    Y_dim = vol.shape[1]
    for y in range(Y_dim):
        tmp_slice = np.zeros_like(vol[:, y, :]).astype(np.float32)
        prev_flow = np.zeros(shape=(shape_of_vol[0], shape_of_vol[2], 2), dtype=np.float32)
        for i in range(kernel.size):
            if i != kernel.size//2:
                flow = get_flow_(padded_vol[:, y + i, :], vol[:, y, :], l, w, prev_flow)
                prev_flow = flow
                if __debug__:
                    min_OF_iter = np.min(flow)
                    if min_OF_iter < min_OF:
                        min_OF = min_OF_iter
                    max_OF_iter = np.max(flow)
                    if max_OF < max_OF_iter:
                        max_OF = max_OF_iter                        
                OF_compensated_slice = warp_slice(padded_vol[:, y + i, :], flow)
                tmp_slice += OF_compensated_slice * kernel[i]
            else:
                tmp_slice += vol[:, y, :] * kernel[i]
        filtered_vol[:, y, :] = tmp_slice
        #logging.info(f"Filtering along Y {int(100*(y/Y_dim))}%")
        __percent__ = int(100*(y/Y_dim))
    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"Filtering along Y spent {time_1 - time_0} seconds")
        logging.debug(f"Min OF val: {min_OF}")
        logging.debug(f"Max OF val: {max_OF}")
    return filtered_vol

def OF_filter_along_Y(vol, kernel, l, w, mean):
    global __percent__
    logging.info(f"Filtering along Y with l={l}, w={w}, and kernel length={kernel.size}")
    if __debug__:
        time_0 = time.perf_counter()
        min_OF = 1000
        max_OF = -1000 
    filtered_vol = np.zeros_like(vol).astype(np.float32)
    shape_of_vol = np.shape(vol)
    #padded_vol = np.zeros(shape=(shape_of_vol[0], shape_of_vol[1] + kernel.size, shape_of_vol[2]))
    padded_vol = np.full(shape=(shape_of_vol[0], shape_of_vol[1] + kernel.size, shape_of_vol[2]), fill_value=mean)
    padded_vol[:, kernel.size//2:shape_of_vol[1] + kernel.size//2, :] = vol
    Y_dim = vol.shape[1]
    #prev_flow = None
    for y in range(Y_dim):
        tmp_slice = np.zeros_like(vol[:, y, :]).astype(np.float32)
        assert kernel.size % 2 != 0 # kernel.size must be odd
        prev_flow = np.zeros(shape=(shape_of_vol[0], shape_of_vol[2], 2), dtype=np.float32)
        for i in range((kernel.size//2) - 1, -1, -1):
            #print(i)
            flow = get_flow(padded_vol[:, y + i, :], vol[:, y, :], l, w, prev_flow)
            prev_flow = flow
            if __debug__:
                min_OF_iter = np.min(flow)
                if min_OF_iter < min_OF:
                    min_OF = min_OF_iter
                max_OF_iter = np.max(flow)
                if max_OF < max_OF_iter:
                    max_OF = max_OF_iter                        
            OF_compensated_slice = warp_slice(padded_vol[:, y + i, :], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        tmp_slice += vol[:, y, :] * kernel[kernel.size//2]
        prev_flow = np.zeros(shape=(shape_of_vol[0], shape_of_vol[2], 2), dtype=np.float32)
        for i in range(kernel.size//2+1, kernel.size):
            #print(i)
            flow = get_flow(padded_vol[:, y + i, :], vol[:, y, :], l, w, prev_flow)
            prev_flow = flow
            if __debug__:
                min_OF_iter = np.min(flow)
                if min_OF_iter < min_OF:
                    min_OF = min_OF_iter
                max_OF_iter = np.max(flow)
                if max_OF < max_OF_iter:
                    max_OF = max_OF_iter                        
            OF_compensated_slice = warp_slice(padded_vol[:, y + i, :], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        filtered_vol[:, y, :] = tmp_slice
        __percent__ = int(100*(y/Y_dim))
    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"Filtering along Y spent {time_1 - time_0} seconds")
        logging.debug(f"Min OF val: {min_OF}")
        logging.debug(f"Max OF val: {max_OF}")
    return filtered_vol

def no_OF_filter_along_Y(vol, kernel, mean):
    global __percent__
    logging.info(f"Filtering along Y with l={l}, w={w}, and kernel length={kernel.size}")
    if __debug__:
        time_0 = time.perf_counter()
    filtered_vol = np.zeros_like(vol).astype(np.float32)
    shape_of_vol = np.shape(vol)
    #padded_vol = np.zeros(shape=(shape_of_vol[0], shape_of_vol[1] + kernel.size, shape_of_vol[2]))
    padded_vol = np.full(shape=(shape_of_vol[0], shape_of_vol[1] + kernel.size, shape_of_vol[2]), fill_value=mean)
    padded_vol[:, kernel.size//2:shape_of_vol[1] + kernel.size//2, :] = vol
    Y_dim = vol.shape[1]
    for y in range(Y_dim):
        tmp_slice = np.zeros_like(vol[:, y, :]).astype(np.float32)
        for i in range(kernel.size):
            tmp_slice += padded_vol[:, y + i, :] * kernel[i]
        filtered_vol[:, y, :] = tmp_slice
        #logging.info(f"Filtering along Y {int(100*(y/Y_dim))}%")
        __percent__ = int(100*(y/Y_dim))
    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"Filtering along Y spent {time_1 - time_0} seconds")
    return filtered_vol

def OF_filter_along_X(vol, kernel, l, w, mean):
    global __percent__
    logging.info(f"Filtering along X with l={l}, w={w}, and kernel length={kernel.size}")
    if __debug__:
        time_0 = time.perf_counter()
        min_OF = 1000
        max_OF = -1000
    filtered_vol = np.zeros_like(vol).astype(np.float32)
    shape_of_vol = np.shape(vol)
    #padded_vol = np.zeros(shape=(shape_of_vol[0], shape_of_vol[1], shape_of_vol[2] + kernel.size))
    padded_vol = np.full(shape=(shape_of_vol[0], shape_of_vol[1], shape_of_vol[2] + kernel.size), fill_value=mean)
    padded_vol[:, :, kernel.size//2:shape_of_vol[2] + kernel.size//2] = vol
    X_dim = vol.shape[2]
    #prev_flow = None
    for x in range(X_dim):
        tmp_slice = np.zeros_like(vol[:, :, x]).astype(np.float32)
        assert kernel.size % 2 != 0 # kernel.size must be odd
        prev_flow = np.zeros(shape=(shape_of_vol[0], shape_of_vol[1], 2), dtype=np.float32)
        for i in range((kernel.size//2) - 1, -1, -1):
            #print(i)
            flow = get_flow(padded_vol[:, :, x + i], vol[:, :, x], l, w, prev_flow)
            prev_flow = flow
            if __debug__:
                min_OF_iter = np.min(flow)
                if min_OF_iter < min_OF:
                    min_OF = min_OF_iter
                max_OF_iter = np.max(flow)
                if max_OF < max_OF_iter:
                    max_OF = max_OF_iter
            OF_compensated_slice = warp_slice(padded_vol[:, :, x + i], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        tmp_slice += vol[:, :, x] * kernel[kernel.size//2]
        prev_flow = np.zeros(shape=(shape_of_vol[0], shape_of_vol[1], 2), dtype=np.float32)
        for i in range(kernel.size//2+1, kernel.size):
            #print(i)
            flow = get_flow(padded_vol[:, :, x + i], vol[:, :, x], l, w, prev_flow)
            prev_flow = flow
            if __debug__:
                min_OF_iter = np.min(flow)
                if min_OF_iter < min_OF:
                    min_OF = min_OF_iter
                max_OF_iter = np.max(flow)
                if max_OF < max_OF_iter:
                    max_OF = max_OF_iter
            OF_compensated_slice = warp_slice(padded_vol[:, :, x + i], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        filtered_vol[:, :, x] = tmp_slice
        __percent__ = int(100*(x/X_dim))
    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"Filtering along X spent {time_1 - time_0} seconds")
    return filtered_vol

def OF_filter_along_X_(vol, kernel, l, w, mean):
    global __percent__
    logging.info(f"Filtering along X with l={l}, w={w}, and kernel length={kernel.size}")
    if __debug__:
        time_0 = time.perf_counter()
    filtered_vol = np.zeros_like(vol).astype(np.float32)
    shape_of_vol = np.shape(vol)
    #padded_vol = np.zeros(shape=(shape_of_vol[0], shape_of_vol[1], shape_of_vol[2] + kernel.size))
    padded_vol = np.full(shape=(shape_of_vol[0], shape_of_vol[1], shape_of_vol[2] + kernel.size), fill_value=mean)
    padded_vol[:, :, kernel.size//2:shape_of_vol[2] + kernel.size//2] = vol
    X_dim = vol.shape[2]
    prev_flow = None
    for x in range(X_dim):
        tmp_slice = np.zeros_like(vol[:, :, x]).astype(np.float32)
        for i in range(kernel.size):
            if i != kernel.size//2:
                flow = get_flow_(padded_vol[:, :, x + i], vol[:, :, x], l, w, prev_flow)
                prev_flow = flow
                OF_compensated_slice = warp_slice(padded_vol[:, :, x + i], flow)
                tmp_slice += OF_compensated_slice * kernel[i]
            else:
                tmp_slice += vol[:, :, x] * kernel[i]
        filtered_vol[:, :, x] = tmp_slice
        #logging.info(f"Filtering along X {int(100*(x/X_dim))}%")
        __percent__ = int(100*(x/X_dim))
    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"Filtering along X spent {time_1 - time_0} seconds")
    return filtered_vol

def no_OF_filter_along_X(vol, kernel, mean):
    global __percent__
    logging.info(f"Filtering along X with l={l}, w={w}, and kernel length={kernel.size}")
    if __debug__:
        time_0 = time.perf_counter()
    filtered_vol = np.zeros_like(vol).astype(np.float32)
    shape_of_vol = np.shape(vol)
    #padded_vol = np.zeros(shape=(shape_of_vol[0], shape_of_vol[1], shape_of_vol[2] + kernel.size))
    padded_vol = np.full(shape=(shape_of_vol[0], shape_of_vol[1], shape_of_vol[2] + kernel.size), fill_value=mean)
    padded_vol[:, :, kernel.size//2:shape_of_vol[2] + kernel.size//2] = vol
    X_dim = vol.shape[2]
    for x in range(X_dim):
        tmp_slice = np.zeros_like(vol[:, :, x]).astype(np.float32)
        for i in range(kernel.size):
            tmp_slice += padded_vol[:, :, x + i] * kernel[i]
        filtered_vol[:, :, x] = tmp_slice
        #logging.info(f"Filtering along X {int(100*(x/X_dim))}%")
        __percent__ = int(100*(x/X_dim))
    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"Filtering along X spent {time_1 - time_0} seconds")
    return filtered_vol

def OF_filter(vol, kernel, l, w):
    mean = vol.mean()
    filtered_vol_Z = OF_filter_along_Z(vol, kernel[0], l, w, mean)
    filtered_vol_ZY = OF_filter_along_Y(filtered_vol_Z, kernel[1], l, w, mean)
    filtered_vol_ZYX = OF_filter_along_X(filtered_vol_ZY, kernel[2], l, w, mean)
    return filtered_vol_ZYX

def no_OF_filter(vol, kernel):
    mean = vol.mean()
    filtered_vol_Z = no_OF_filter_along_Z(vol, kernel[0], mean)
    filtered_vol_ZY = no_OF_filter_along_Y(filtered_vol_Z, kernel[1], mean)
    filtered_vol_ZYX = no_OF_filter_along_X(filtered_vol_ZY, kernel[2], mean)
    return filtered_vol_ZYX

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
parser.add_argument("-w", "--winsize", type=int_or_str,
                    help="Size of the window used by the optical flow estimator",
                    default=OF_WINDOW_SIZE)
parser.add_argument("-v", "--verbosity", type=int_or_str,
                    help="Verbosity level", default=0)
parser.add_argument("-n", "--no_OF", action="store_true", help="Disable optical flow compensation")
parser.add_argument("-m", "--memory_map", action="store_true", help="Enable memory-mapping (see https://mrcfile.readthedocs.io/en/stable/usage_guide.html#dealing-with-large-files, only for MRC files)")

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
    w = args.winsize
    
    #logging.debug(f"Using transpose pattern {args.transpose} {type(args.transpose)}")
    #transpose_pattern = tuple([int(i) for i in args.transpose])
    #logging.info(f"transpose={transpose_pattern}")

    if __debug__:
        logging.info(f"reading \"{args.input}\"")
        time_0 = time.perf_counter()

    logging.debug(f"input = {args.input}")

    MRC_input = ( args.input.split('.')[-1] == "MRC" or args.input.split('.')[-1] == "mrc" )
    if MRC_input:
        if args.memory_map:
            logging.info(f"Using memory mapping")
            vol_MRC = rc = mrcfile.mmap(args.input, mode='r+')
        else:
            vol_MRC = mrcfile.open(args.input)
        vol = vol_MRC.data
    else:
        vol = skimage.io.imread(args.input, plugin="tifffile").astype(np.float32)

    logging.info(f"shape of the input volume (Z, Y, X) = {vol.shape}")
    logging.info(f"type of the volume = {vol.dtype}")

    #vol = np.transpose(vol, transpose_pattern)
    #logging.info(f"shape of the volume to denoise (Z, Y, X) = {vol.shape}")

    if __debug__:
        time_1 = time.perf_counter()
        logging.info(f"read \"{args.input}\" in {time_1 - time_0} seconds")

    logging.info(f"{args.input} type = {vol.dtype}")
    logging.info(f"{args.input} max = {vol.max()}")
    logging.info(f"{args.input} min = {vol.min()}")
    logging.info(f"Input vol average = {vol.mean()}")

    kernel = [None]*3
    kernel[0] = get_gaussian_kernel(sigma[0])
    kernel[1] = get_gaussian_kernel(sigma[1])
    kernel[2] = get_gaussian_kernel(sigma[2])
    logging.info(f"length of each filter (Z, Y, X) = {[len(i) for i in [*kernel]]}")
    if args.no_OF:
        filtered_vol = no_OF_filter(vol, kernel)
    else:
        filtered_vol = OF_filter(vol, kernel, l, w)

    #filtered_vol = np.transpose(filtered_vol, transpose_pattern)
    logging.info(f"shape of the denoised volume (Z, Y, X) = {filtered_vol.shape}")

    logging.info(f"{args.output} type = {filtered_vol.dtype}")
    logging.info(f"{args.output} max = {filtered_vol.max()}")
    logging.info(f"{args.output} min = {filtered_vol.min()}")
    logging.info(f"Output vol average = {filtered_vol.mean()}")
    
    if __debug__:
        logging.info(f"writting \"{args.output}\"")
        time_0 = time.perf_counter()

    logging.debug(f"output = {args.output}")
        
    MRC_output = ( args.output.split('.')[-1] == "MRC" or args.output.split('.')[-1] == "mrc" )

    if MRC_output:
        logging.debug(f"Writting MRC file")
        with mrcfile.new(args.output, overwrite=True) as mrc:
            mrc.set_data(filtered_vol.astype(np.float32))
            mrc.data
    else:
        if np.max(filtered_vol) < 256:
            logging.debug(f"Writting TIFF file (uint8)")
            skimage.io.imsave(args.output, filtered_vol.astype(np.uint8), plugin="tifffile")
        else:
            logging.debug(f"Writting TIFF file (uint16)")
            skimage.io.imsave(args.output, filtered_vol.astype(np.uint16), plugin="tifffile")

    if __debug__:
        time_1 = time.perf_counter()        
        logging.info(f"written \"{args.output}\" in {time_1 - time_0} seconds")
