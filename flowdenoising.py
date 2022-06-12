#!/usr/bin/env python
'''3D Gaussian filtering controlled by the optical flow.
'''

# "flowdenoising.py" is part of "https://github.com/microscopy-processing/FlowDenoising", authored by:
# * J.J. Fernández (CSIC).
# * V. González-Ruiz (UAL).

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

LOGGING_FORMAT = "[%(asctime)s] (%(levelname)s) %(message)s"

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
OF_LEVELS = 3
OF_WINDOW_SIDE = 5
OF_ITERS = 3
OF_POLY_N = 5
OF_POLY_SIGMA = 1.2
SIGMA = 2.0

def warp_slice(reference, flow):
    if __debug__:
        logging.debug("Warping slice")
        time_0 = time.process_time()
    height, width = flow.shape[:2]
    map_x = np.tile(np.arange(width), (height, 1))
    map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
    map_xy = (flow + np.dstack((map_x, map_y))).astype('float32')
    warped_slice = cv2.remap(reference, map_xy, None, interpolation=cv2.INTER_LINEAR, borderMode=OFCA_EXTENSION_MODE)
    if __debug__:
        time_1 = time.process_time()
        logging.debug(f"Slice warped in {time_1 - time_0} seconds")
    return warped_slice

def get_flow(reference, target, l=OF_LEVELS, w=OF_WINDOW_SIDE):
    if __debug__:
        logging.debug("Computing OF")
        time_0 = time.process_time()
    flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=None, pyr_scale=0.5, levels=l, winsize=w, iterations=OF_ITERS, poly_n=OF_POLY_N, poly_sigma=OF_POLY_SIGMA, flags=0)
    if __debug__:
        time_1 = time.process_time()
        logging.debug(f"OF computed in {time_1 - time_0} seconds")
    return flow

def OF_filter_along_Z(stack, kernel, l, w):
    logging.info(f"Filtering along Z with l={l} and w={w}")
    if __debug__:
        time_0 = time.process_time()
    filtered_stack = np.zeros_like(stack).astype(np.float32)
    shape_of_stack = np.shape(stack)
    padded_stack = np.zeros(shape=(shape_of_stack[0] + kernel.size, shape_of_stack[1], shape_of_stack[2]))
    padded_stack[kernel.size//2:shape_of_stack[0] + kernel.size//2, :, :] = stack
    Z_dim = stack.shape[0]
    for z in range(Z_dim):
        tmp_slice = np.zeros_like(stack[z]).astype(np.float32)
        for i in range(kernel.size):
            if i != kernel.size//2:
                flow = get_flow(padded_stack[z + i], stack[z], l, w)
                OF_compensated_slice = warp_slice(padded_stack[z + i], flow)
                tmp_slice += OF_compensated_slice * kernel[i]
            else:
                tmp_slice += stack[z, :, :] * kernel[i]
        filtered_stack[z, :, :] = tmp_slice
        logging.info(f"Filtering along Z {int(100*(z/Z_dim))}%")
    if __debug__:
        time_1 = time.process_time()
        logging.debug(f"Filtering along Z spent {time_1 - time_0} seconds")
    return filtered_stack

def no_OF_filter_along_Z(stack, kernel):
    logging.info(f"Filtering along Z with l={l} and w={w}")
    if __debug__:
        time_0 = time.process_time()
    filtered_stack = np.zeros_like(stack).astype(np.float32)
    shape_of_stack = np.shape(stack)
    padded_stack = np.zeros(shape=(shape_of_stack[0] + kernel.size, shape_of_stack[1], shape_of_stack[2]))
    padded_stack[kernel.size//2:shape_of_stack[0] + kernel.size//2, ...] = stack
    Z_dim = stack.shape[0]
    for z in range(Z_dim):
        tmp_slice = np.zeros_like(stack[z, :, :]).astype(np.float32)
        for i in range(kernel.size):
            tmp_slice += padded_stack[z + i, :, :] * kernel[i]
        filtered_stack[z, :, :] = tmp_slice
        logging.info(f"Filtering along Z {int(100*(z/Z_dim))}%")
    if __debug__:
        time_1 = time.process_time()
        logging.debug(f"Filtering along Z spent {time_1 - time_0} seconds")
    return filtered_stack

def OF_filter_along_Y(stack, kernel, l, w):
    logging.info(f"Filtering along Y with l={l} and w={w}")
    if __debug__:
        time_0 = time.process_time()
    filtered_stack = np.zeros_like(stack).astype(np.float32)
    shape_of_stack = np.shape(stack)
    padded_stack = np.zeros(shape=(shape_of_stack[0], shape_of_stack[1] + kernel.size, shape_of_stack[2]))
    padded_stack[:, kernel.size//2:shape_of_stack[1] + kernel.size//2, :] = stack
    Y_dim = stack.shape[1]
    for y in range(Y_dim):
        tmp_slice = np.zeros_like(stack[:, y, :]).astype(np.float32)
        for i in range(kernel.size):
            if i != kernel.size//2:
                flow = get_flow(padded_stack[:, y + i, :], stack[:, y, :], l, w)
                OF_compensated_slice = warp_slice(padded_stack[:, y + i, :], flow)
                tmp_slice += OF_compensated_slice * kernel[i]
            else:
                tmp_slice += stack[:, y, :] * kernel[i]
        filtered_stack[:, y, :] = tmp_slice
        logging.info(f"Filtering along Y {int(100*(y/Y_dim))}%")
    if __debug__:
        time_1 = time.process_time()
        logging.debug(f"Filtering along Y spent {time_1 - time_0} seconds")
    return filtered_stack

def no_OF_filter_along_Y(stack, kernel):
    logging.info(f"Filtering along Y with l={l} and w={w}")
    if __debug__:
        time_0 = time.process_time()
    filtered_stack = np.zeros_like(stack).astype(np.float32)
    shape_of_stack = np.shape(stack)
    padded_stack = np.zeros(shape=(shape_of_stack[0], shape_of_stack[1] + kernel.size, shape_of_stack[2]))
    padded_stack[:, kernel.size//2:shape_of_stack[1] + kernel.size//2, :] = stack
    Y_dim = stack.shape[1]
    for y in range(Y_dim):
        tmp_slice = np.zeros_like(stack[:, y, :]).astype(np.float32)
        for i in range(kernel.size):
            tmp_slice += padded_stack[:, y + i, :] * kernel[i]
        filtered_stack[:, y, :] = tmp_slice
        logging.info(f"Filtering along Y {int(100*(y/Y_dim))}%")
    if __debug__:
        time_1 = time.process_time()
        logging.debug(f"Filtering along Y spent {time_1 - time_0} seconds")
    return filtered_stack

def OF_filter_along_X(stack, kernel, l, w):
    logging.info(f"Filtering along X with l={l} and w={w}")
    if __debug__:
        time_0 = time.process_time()
    filtered_stack = np.zeros_like(stack).astype(np.float32)
    shape_of_stack = np.shape(stack)
    padded_stack = np.zeros(shape=(shape_of_stack[0], shape_of_stack[1], shape_of_stack[2] + kernel.size))
    padded_stack[:, :, kernel.size//2:shape_of_stack[2] + kernel.size//2] = stack
    X_dim = stack.shape[2]
    for x in range(X_dim):
        tmp_slice = np.zeros_like(stack[:, :, x]).astype(np.float32)
        for i in range(kernel.size):
            if i != kernel.size//2:
                flow = get_flow(padded_stack[:, :, x + i], stack[:, :, x], l, w)
                OF_compensated_slice = warp_slice(padded_stack[:, :, x + i], flow)
                tmp_slice += OF_compensated_slice * kernel[i]
            else:
                tmp_slice += stack[:, :, x] * kernel[i]
        filtered_stack[:, :, x] = tmp_slice
        logging.info(f"Filtering along X {int(100*(x/X_dim))}%")
    if __debug__:
        time_1 = time.process_time()
        logging.debug(f"Filtering along X spent {time_1 - time_0} seconds")
    return filtered_stack

def no_OF_filter_along_X(stack, kernel):
    logging.info(f"Filtering along X with l={l} and w={w}")
    if __debug__:
        time_0 = time.process_time()
    filtered_stack = np.zeros_like(stack).astype(np.float32)
    shape_of_stack = np.shape(stack)
    padded_stack = np.zeros(shape=(shape_of_stack[0], shape_of_stack[1], shape_of_stack[2] + kernel.size))
    padded_stack[:, :, kernel.size//2:shape_of_stack[2] + kernel.size//2] = stack
    X_dim = stack.shape[2]
    for x in range(X_dim):
        tmp_slice = np.zeros_like(stack[:, :, x]).astype(np.float32)
        for i in range(kernel.size):
            tmp_slice += padded_stack[:, :, x + i] * kernel[i]
        filtered_stack[:, :, x] = tmp_slice
        logging.info(f"Filtering along X {int(100*(x/X_dim))}%")
    if __debug__:
        time_1 = time.process_time()
        logging.debug(f"Filtering along X spent {time_1 - time_0} seconds")
    return filtered_stack

def OF_filter(stack, kernel, l, w):
    filtered_stack_Z = OF_filter_along_Z(stack, kernel, l, w)
    filtered_stack_ZY = OF_filter_along_Y(filtered_stack_Z, kernel, l, w)
    filtered_stack_ZYX = OF_filter_along_X(filtered_stack_ZY, kernel, l, w)
    return filtered_stack_ZYX

def no_OF_filter(stack, kernel):
    filtered_stack_Z = no_OF_filter_along_Z(stack, kernel)
    filtered_stack_ZY = no_OF_filter_along_Y(filtered_stack_Z, kernel)
    filtered_stack_ZYX = no_OF_filter_along_X(filtered_stack_ZY, kernel)
    return filtered_stack_ZYX

def int_or_str(text):
    '''Helper function for argument parsing.'''
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument("-f", "--format", type=int_or_str,
#                    help="Input and output file format (MRC or TIFF)",
#                    default="MRC")
parser.add_argument("-i", "--input", type=int_or_str,
                    help="Input a MRC-file or a multi-image TIFF-file",
                    default="./stack.tif")
parser.add_argument("-o", "--output", type=int_or_str,
                    help="Output a MRC-file or a multi-image TIFF-file",
                    default="./filtered_stack.tif")
#parser.add_argument("-n", "--number_of_images", type=int_or_str,
#                    help="Number of input images (only if the sequence of images is input)",
#                    default=32)
parser.add_argument("-s", "--sigma", type=np.float32,
                    help="Gaussian sigma",
                    default=SIGMA)
parser.add_argument("-l", "--levels", type=int_or_str,
                    help="Number of levels of the Gaussian pyramid used by the optical flow estimator",
                    default=OF_LEVELS)
parser.add_argument("-w", "--winside", type=int_or_str,
                    help="Side of the window used by the optical flow estimator",
                    default=OF_WINDOW_SIDE)
parser.add_argument("-v", "--verbosity", type=int_or_str,
                    help="Verbosity level", default=0)
parser.add_argument("-n", "--no_OF", action="store_true", help="Disable Optical Flow compensation")

if __name__ == "__main__":
    parser.description = __doc__
    args = parser.parse_known_args()[0]
    if args.verbosity == 2:
        logging.basicConfig(format=LOGGING_FORMAT, level=logging.DEBUG)
        logging.info("Verbosity level = 2")
    elif args.verbosity == 1:
        logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)
        logging.info("Verbosity level = 1")        
    else:
        logging.basicConfig(format=LOGGING_FORMAT, level=logging.CRITICAL)

    sigma = args.sigma
    l = args.levels
    w = args.winside

    if __debug__:
        logging.info(f"reading \"{args.input}\"")
        time_0 = time.process_time()

    logging.debug(f"input = {args.input}")

    MRC_input = ( args.input.split('.')[1] == "MRC" or args.input.split('.')[1] == "mrc" )
    if MRC_input:
        stack_MRC = mrcfile.open(args.input)
        stack = stack_MRC.data
    else:
        stack = skimage.io.imread(args.input, plugin="tifffile").astype(np.float32)

    if __debug__:
        time_1 = time.process_time()
        logging.info(f"read \"{args.input}\" in {time_1 - time_0} seconds")

    logging.info(f"{args.input} type = {stack.dtype}")
    logging.info(f"{args.input} max = {stack.max()}")
    logging.info(f"{args.input} min = {stack.min()}")
    logging.info(f"Input stack average = {stack.mean()}")
    
    kernel = get_gaussian_kernel(sigma)
    if args.no_OF:
        filtered_stack = no_OF_filter(stack, kernel)
    else:
        filtered_stack = OF_filter(stack, kernel, l, w)

    logging.info(f"{args.output} type = {filtered_stack.dtype}")
    logging.info(f"{args.output} max = {filtered_stack.max()}")
    logging.info(f"{args.output} min = {filtered_stack.min()}")
    logging.info(f"Output stack average = {filtered_stack.mean()}")
    
    if __debug__:
        logging.info(f"writting \"{args.output}\"")
        time_0 = time.process_time()

    logging.debug(f"output = {args.output}")
        
    MRC_output = ( args.output.split('.')[1] == "MRC" or args.output.split('.')[1] == "mrc" )

    if MRC_output:
        with mrcfile.new(args.output, overwrite=True) as mrc:
            mrc.set_data(filtered_stack.astype(np.float32))
            mrc.data
    else:
        skimage.io.imsave(args.output, filtered_stack, plugin="tifffile")

    if __debug__:
        time_1 = time.process_time()        
        logging.info(f"written \"{args.output}\" in {time_1 - time_0} seconds")
