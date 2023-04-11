#!/usr/bin/env python
'''3D Gaussian filtering controlled by the optical flow.
'''

#
# "flowdenoising_GPU.py" is part of
# "https://github.com/microscopy-processing/FlowDenoising", authored
# by:
#
# * J.J. Fernández (CSIC).
# * V. González-Ruiz (UAL).
#
# This code implements multiple-processing Gaussian filtering of 3D
# data, and if a GPU is detected, the optical flow is estimated in it.
#
# Please, refer to the LICENSE.txt to know the terms of usage of this
# software.
#

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
from shared_code import *

import concurrent
import multiprocessing
from multiprocessing import shared_memory, Value
from concurrent.futures.process import ProcessPoolExecutor

LOGGING_FORMAT = "[%(asctime)s] (%(levelname)s) %(message)s"

__percent__ = Value('f', 0)

def warp_slice(reference, flow):
    height, width = flow.shape[:2]
    map_x = np.tile(np.arange(width), (height, 1))
    map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
    map_xy = (flow + np.dstack((map_x, map_y))).astype('float32')
    warped_slice = cv2.remap(reference, map_xy, None, interpolation=cv2.INTER_LINEAR, borderMode=OFCA_EXTENSION_MODE)
    return warped_slice

def get_flow_GPU(reference, target, l=OF_LEVELS, w=OF_WINDOW_SIZE, prev_flow=None):
    if __debug__:
        time_0 = time.perf_counter()
    GPU_target = cv2.cuda_GpuMat()
    GPU_target.upload(target)
    GPU_reference = cv2.cuda_GpuMat()
    GPU_reference.upload(reference)
    GPU_prev_flow = cv2.cuda_GpuMat()
    GPU_prev_flow.upload(prev_flow)

    # create optical flow instance
    flower = cv2.cuda_FarnebackOpticalFlow.create(numLevels=l, pyrScale=0.5, fastPyramids=False, winSize=w, numIters=OF_ITERS, polyN=OF_POLY_N, polySigma=OF_POLY_SIGMA, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    #flower = cv2.cuda_FarnebackOpticalFlow.create(numLevels=l, pyrScale=0.5, fastPyramids=False, winSize=w, numIters=OF_ITERS, polyN=OF_POLY_N, polySigma=OF_POLY_SIGMA, flags=0)
    
    # calculate optical flow
    #gpu_flow = cv2.cuda.FarnebackOpticalFlow.calc(flower, I0=gpu_target, I1=gpu_reference, flow=None)
    GPU_flow = cv2.cuda.FarnebackOpticalFlow.calc(flower, I0=GPU_target, I1=GPU_reference, flow=GPU_prev_flow)

    flow = GPU_flow.download()
    
    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"OF computed in {1000*(time_1 - time_0):4.3f} ms, max_X={np.max(flow[0]):+3.2f}, min_X={np.min(flow[0]):+3.2f}, max_Y={np.max(flow[1]):+3.2f}, min_Y={np.min(flow[1]):+3.2f}")
    return flow

def get_flow_CPU(reference, target, l=OF_LEVELS, w=OF_WINDOW_SIZE, prev_flow=None):
    if __debug__:
        time_0 = time.perf_counter()
    flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=prev_flow, pyr_scale=0.5, levels=l, winsize=w, iterations=OF_ITERS, poly_n=OF_POLY_N, poly_sigma=OF_POLY_SIGMA, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    #flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=None, pyr_scale=0.5, levels=l, winsize=w, iterations=OF_ITERS, poly_n=OF_POLY_N, poly_sigma=OF_POLY_SIGMA, flags=0)
    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"OF computed in {1000*(time_1 - time_0):4.3f} ms, max_X={np.max(flow[0]):+3.2f}, min_X={np.min(flow[0]):+3.2f}, max_Y={np.max(flow[1]):+3.2f}, min_Y={np.min(flow[1]):+3.2f}")
    return flow

get_flow = get_flow_GPU

def OF_filter_along_Z_slice(z, kernel):
    ks2 = kernel.size//2
    tmp_slice = np.zeros_like(vol[z, :, :]).astype(np.float32)
    assert kernel.size % 2 != 0 # kernel.size must be odd
    prev_flow = np.zeros(shape=(vol.shape[1], vol.shape[2], 2), dtype=np.float32)
    for i in range(ks2 - 1, -1, -1):
        flow = get_flow(vol[(z + i - ks2) % vol.shape[0], :, :],
                        vol[z, :, :], l, w, prev_flow)
        prev_flow = flow
        OF_compensated_slice = warp_slice(vol[(z + i - ks2) % vol.shape[0], :, :], flow)
        tmp_slice += OF_compensated_slice * kernel[i]
    tmp_slice += vol[z, :, :] * kernel[ks2]
    prev_flow = np.zeros(shape=(vol.shape[1], vol.shape[2], 2), dtype=np.float32)
    for i in range(ks2 + 1, kernel.size):
        flow = get_flow(vol[(z + i - ks2) % vol.shape[0], :, :],
                        vol[z, :, :], l, w, prev_flow)
        prev_flow = flow
        OF_compensated_slice = warp_slice(vol[(z + i - ks2) % vol.shape[0], :, :], flow)
        tmp_slice += OF_compensated_slice * kernel[i]
    filtered_vol[z, :, :] = tmp_slice
    __percent__.value += 1

def OF_filter_along_Y_slice(y, kernel):
    ks2 = kernel.size//2
    tmp_slice = np.zeros_like(vol[:, y, :]).astype(np.float32)
    assert kernel.size % 2 != 0 # kernel.size must be odd
    prev_flow = np.zeros(shape=(vol.shape[0], vol.shape[2], 2), dtype=np.float32)
    for i in range(ks2 - 1, -1, -1):
        flow = get_flow(vol[:, (y + i - ks2) % vol.shape[1], :],
                        vol[:, y, :], l, w, prev_flow)
        prev_flow = flow
        OF_compensated_slice = warp_slice(vol[:, (y + i - ks2) % vol.shape[1], :], flow)
        tmp_slice += OF_compensated_slice * kernel[i]
    tmp_slice += vol[:, y, :] * kernel[ks2]
    prev_flow = np.zeros(shape=(vol.shape[0], vol.shape[2], 2), dtype=np.float32)
    for i in range(ks2 + 1, kernel.size):
        flow = get_flow(vol[:, (y + i - ks2) % vol.shape[1], :],
                        vol[:, y, :], l, w, prev_flow)
        prev_flow = flow
        OF_compensated_slice = warp_slice(vol[:, (y + i - ks2) % vol.shape[1], :], flow)
        tmp_slice += OF_compensated_slice * kernel[i]
    filtered_vol[:, y, :] = tmp_slice
    __percent__.value += 1

def OF_filter_along_X_slice(x, kernel):
    ks2 = kernel.size//2
    tmp_slice = np.zeros_like(vol[:, :, x]).astype(np.float32)
    assert kernel.size % 2 != 0 # kernel.size must be odd
    prev_flow = np.zeros(shape=(vol.shape[0], vol.shape[1], 2), dtype=np.float32)
    for i in range(ks2 - 1, -1, -1):
        flow = get_flow(vol[:, :, (x + i - ks2) % vol.shape[2]],
                        vol[:, :, x], l, w, prev_flow)
        prev_flow = flow
        OF_compensated_slice = warp_slice(vol[:, :, (x + i - ks2) % vol.shape[2]], flow)
        tmp_slice += OF_compensated_slice * kernel[i]
    tmp_slice += vol[:, :, x] * kernel[ks2]
    prev_flow = np.zeros(shape=(vol.shape[0], vol.shape[1], 2), dtype=np.float32)
    for i in range(ks2 + 1, kernel.size):
        flow = get_flow(vol[:, :, (x + i - ks2) % vol.shape[2]],
                        vol[:, :, x], l, w, prev_flow)
        prev_flow = flow
        OF_compensated_slice = warp_slice(vol[:, :, (x + i - ks2) % vol.shape[2]], flow)
        tmp_slice += OF_compensated_slice * kernel[i]
    filtered_vol[:, :, x] = tmp_slice
    __percent__.value += 1

def OF_filter_along_Z_chunk(chunk_index, chunk_size, chunk_offset, kernel):
    #cv2.cuda.setDevice(chunk_index)
    for z in range(chunk_size):
        OF_filter_along_Z_slice(chunk_index*chunk_size + z + chunk_offset, kernel)
    return chunk_index

def OF_filter_along_Y_chunk(chunk_index, chunk_size, chunk_offset, kernel):
    for y in range(chunk_size):
        OF_filter_along_Y_slice(chunk_index*chunk_size + y + chunk_offset, kernel)
    return chunk_index

def OF_filter_along_X_chunk(chunk_index, chunk_size, chunk_offset, kernel):
    for x in range(chunk_size):
        OF_filter_along_X_slice(chunk_index*chunk_size + x + chunk_offset, kernel)
    return chunk_index
    
def OF_filter_along_Z(kernel, l, w):
    global __percent__
    logging.info(f"Filtering along Z with l={l}, w={w}, and kernel length={kernel.size}")

    if __debug__:
        time_0 = time.perf_counter()
        min_OF = 1000
        max_OF = -1000

    Z_dim = vol.shape[0]
    chunk_size = Z_dim//number_of_processes
    #for i in range(number_of_processes):
    #    OF_filter_along_Z_chunk(i, padded_vol, kernel)
    chunk_indexes = [i for i in range(number_of_processes)]
    chunk_sizes = [chunk_size]*number_of_processes
    chunk_offsets = [0]*number_of_processes
    kernels = [kernel]*number_of_processes
    with ProcessPoolExecutor(max_workers=number_of_processes) as executor:
        for _ in executor.map(OF_filter_along_Z_chunk,
                              chunk_indexes,
                              chunk_sizes,
                              chunk_offsets,
                              kernels):
            logging.debug(f"PE #{_} has finished")
    remainding_slices = Z_dim % number_of_processes
    if remainding_slices > 0:
        chunk_indexes = [i for i in range(remainding_slices)]
        chunk_sizes = [1]*remainding_slices
        chunk_offsets = [chunk_size*number_of_processes]*remainding_slices
        kernels = [kernel]*remainding_slices
        with ProcessPoolExecutor(max_workers=remainding_slices) as executor:
            for _ in executor.map(OF_filter_along_Z_chunk,
                                  chunk_indexes,
                                  chunk_sizes,
                                  chunk_offsets,
                                  kernels):
                logging.debug(f"PU #{_} finished")
    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"Filtering along Z spent {time_1 - time_0} seconds")
        logging.debug(f"Min OF val: {min_OF}")
        logging.debug(f"Max OF val: {max_OF}")

def OF_filter_along_Y(kernel, l, w):
    global __percent__
    logging.info(f"Filtering along Y with l={l}, w={w}, and kernel length={kernel.size}")
    if __debug__:
        time_0 = time.perf_counter()
        min_OF = 1000
        max_OF = -1000

    Y_dim = vol.shape[1]
    chunk_size = Y_dim//number_of_processes
    #for i in range(number_of_processes):
    #    OF_filter_along_Y_chunk(i, padded_vol, kernel)
    chunk_indexes = [i for i in range(number_of_processes)]
    chunk_sizes = [chunk_size]*number_of_processes
    chunk_offsets = [0]*number_of_processes
    kernels = [kernel]*number_of_processes
    with ProcessPoolExecutor(max_workers=number_of_processes) as executor:
        for _ in executor.map(OF_filter_along_Y_chunk,
                              chunk_indexes,
                              chunk_sizes,
                              chunk_offsets,
                              kernels):
            logging.debug(f"PE #{_} has finished")
    remainding_slices = Y_dim % number_of_processes
    if remainding_slices > 0:
        chunk_indexes = [i for i in range(remainding_slices)]
        chunk_sizes = [1]*remainding_slices
        chunk_offsets = [chunk_size*number_of_processes]*remainding_slices
        kernels = [kernel]*remainding_slices
        with ProcessPoolExecutor(max_workers=remainding_slices) as executor:
            for _ in executor.map(OF_filter_along_Y_chunk,
                                  chunk_indexes,
                                  chunk_sizes,
                                  chunk_offsets,
                                  kernels):
                logging.debug(f"PU #{_} finished")

    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"Filtering along Y spent {time_1 - time_0} seconds")
        logging.debug(f"Min OF val: {min_OF}")
        logging.debug(f"Max OF val: {max_OF}")

def OF_filter_along_X(kernel, l, w):
    global __percent__
    logging.info(f"Filtering along X with l={l}, w={w}, and kernel length={kernel.size}")
    if __debug__:
        time_0 = time.perf_counter()
        min_OF = 1000
        max_OF = -1000

    X_dim = vol.shape[2]
    chunk_size = X_dim//number_of_processes
    #for i in range(number_of_processes):
    #    OF_filter_along_X_chunk(i, padded_vol, kernel)
    chunk_indexes = [i for i in range(number_of_processes)]
    chunk_sizes = [chunk_size]*number_of_processes
    chunk_offsets = [0]*number_of_processes
    kernels = [kernel]*number_of_processes
    with ProcessPoolExecutor(max_workers=number_of_processes) as executor:
        for _ in executor.map(OF_filter_along_X_chunk,
                              chunk_indexes,
                              chunk_sizes,
                              chunk_offsets,
                              kernels):
            logging.debug(f"PE #{_} has finished")
    remainding_slices = X_dim % number_of_processes
    if remainding_slices > 0:
        chunk_indexes = [i for i in range(remainding_slices)]
        chunk_sizes = [1]*remainding_slices
        chunk_offsets = [chunk_size*number_of_processes]*remainding_slices
        kernels = [kernel]*remainding_slices
        with ProcessPoolExecutor(max_workers=remainding_slices) as executor:
            for _ in executor.map(OF_filter_along_X_chunk,
                                  chunk_indexes,
                                  chunk_sizes,
                                  chunk_offsets,
                                  kernels):
                logging.debug(f"PU #{_} finished")

    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"Filtering along X spent {time_1 - time_0} seconds")

def OF_filter(kernels, l, w):
    OF_filter_along_Z(kernels[0], l, w)
    vol[...] = filtered_vol[...]
    OF_filter_along_Y(kernels[1], l, w)
    vol[...] = filtered_vol[...]
    OF_filter_along_X(kernels[2], l, w)

def no_OF_filter_along_Z_slice(z, kernel):
    ks2 = kernel.size//2
    tmp_slice = np.zeros(shape=(vol.shape[1], vol.shape[2]), dtype=np.float32)
    for i in range(kernel.size):
        tmp_slice += vol[(z + i - ks2) % vol.shape[0], :, :]*kernel[i]
    filtered_vol[z, :, :] = tmp_slice
    __percent__.value += 1

def no_OF_filter_along_Y_slice(y, kernel):
    ks2 = kernel.size//2
    tmp_slice = np.zeros(shape=(vol.shape[0], vol.shape[2]), dtype=np.float32)
    for i in range(kernel.size):
        tmp_slice += vol[:, (y + i - ks2) % vol.shape[1], :]*kernel[i]
    filtered_vol[:, y, :] = tmp_slice
    __percent__.value += 1

def no_OF_filter_along_X_slice(x, kernel):
    ks2 = kernel.size//2
    tmp_slice = np.zeros(shape=(vol.shape[0], vol.shape[1]), dtype=np.float32)
    for i in range(kernel.size):
        tmp_slice += vol[:, :, (x + i - ks2) % vol.shape[2]]*kernel[i]
    filtered_vol[:, :, x] = tmp_slice
    __percent__.value += 1

def no_OF_filter_along_Z_chunk(chunk_index, chunk_size, chunk_offset, kernel):
    for z in range(chunk_size):
        no_OF_filter_along_Z_slice(chunk_index*chunk_size + z + chunk_offset, kernel)
    return chunk_index

def no_OF_filter_along_Y_chunk(chunk_index, chunk_size, chunk_offset, kernel):
    for y in range(chunk_size):
        no_OF_filter_along_Y_slice(chunk_index*chunk_size + y + chunk_offset, kernel)
    return chunk_index

def no_OF_filter_along_X_chunk(chunk_index, chunk_size, chunk_offset, kernel):
    for x in range(chunk_size):
        no_OF_filter_along_X_slice(chunk_index*chunk_size + x + chunk_offset, kernel)
    return chunk_index

def no_OF_filter_along_Z(kernel):
    logging.info(f"Filtering along Z with kernel length={kernel.size}")

    if __debug__:
        time_0 = time.perf_counter()

    Z_dim = vol.shape[0]
    chunk_size = Z_dim//number_of_processes
    #for i in range(number_of_processes):
    #    no_OF_filter_along_Z_chunk(i, kernel)
    chunk_indexes = [i for i in range(number_of_processes)]
    chunk_sizes = [chunk_size]*number_of_processes
    chunk_offsets = [0]*number_of_processes
    kernels = [kernel]*number_of_processes
    with ProcessPoolExecutor(max_workers=number_of_processes) as executor:
        for _ in executor.map(no_OF_filter_along_Z_chunk,
                              chunk_indexes,
                              chunk_sizes,
                              chunk_offsets,
                              kernels):
            logging.debug(f"PU #{_} finished")
    remainding_slices = Z_dim % number_of_processes
    if remainding_slices > 0:
        chunk_indexes = [i for i in range(remainding_slices)]
        chunk_sizes = [1]*remainding_slices
        chunk_offsets = [chunk_size*number_of_processes]*remainding_slices
        kernels = [kernel]*remainding_slices
        with ProcessPoolExecutor(max_workers=remainding_slices) as executor:
            for _ in executor.map(no_OF_filter_along_Z_chunk,
                                  chunk_indexes,
                                  chunk_sizes,
                                  chunk_offsets,
                                  kernels):
                logging.debug(f"PU #{_} finished")

    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"Filtering along Z spent {time_1 - time_0} seconds")

def no_OF_filter_along_Y(kernel):
    logging.info(f"Filtering along Y with kernel length={kernel.size}")

    if __debug__:
        time_0 = time.perf_counter()

    Y_dim = vol.shape[1]
    chunk_size = Y_dim//number_of_processes
    #for i in range(number_of_processes):
    #    no_OF_filter_along_Y_chunk(i, kernel)
    chunk_indexes = [i for i in range(number_of_processes)]
    chunk_sizes = [chunk_size]*number_of_processes
    chunk_offsets = [0]*number_of_processes
    kernels = [kernel]*number_of_processes
    with ProcessPoolExecutor(max_workers=number_of_processes) as executor:
        for _ in executor.map(no_OF_filter_along_Y_chunk,
                              chunk_indexes,
                              chunk_sizes,
                              chunk_offsets,
                              kernels):
            logging.debug(f"PU #{_} finished")
    remainding_slices = Y_dim % number_of_processes
    if remainding_slices > 0:
        chunk_indexes = [i for i in range(remainding_slices)]
        chunk_sizes = [1]*remainding_slices
        chunk_offsets = [chunk_size*number_of_processes]*remainding_slices
        kernels = [kernel]*remainding_slices
        with ProcessPoolExecutor(max_workers=remainding_slices) as executor:
            for _ in executor.map(no_OF_filter_along_Y_chunk,
                                  chunk_indexes,
                                  chunk_sizes,
                                  chunk_offsets,
                                  kernels):
                logging.debug(f"PU #{_} finished")

    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"Filtering along Y spent {time_1 - time_0} seconds")

def no_OF_filter_along_X(kernel):
    logging.info(f"Filtering along X with kernel length={kernel.size}")
    if __debug__:
        time_0 = time.perf_counter()

    X_dim = vol.shape[2]
    chunk_size = X_dim//number_of_processes
    #for i in range(number_of_processes):
    #    no_OF_filter_along_X_chunk(i, kernel)
    chunk_indexes = [i for i in range(number_of_processes)]
    chunk_sizes = [chunk_size]*number_of_processes
    chunk_offsets = [0]*number_of_processes
    kernels = [kernel]*number_of_processes
    with ProcessPoolExecutor(max_workers=number_of_processes) as executor:
        for _ in executor.map(no_OF_filter_along_X_chunk,
                              chunk_indexes,
                              chunk_sizes,
                              chunk_offsets,
                              kernels):
            logging.debug(f"PU #{_} finished")
    remainding_slices = X_dim % number_of_processes
    if remainding_slices > 0:
        chunk_indexes = [i for i in range(remainding_slices)]
        chunk_sizes = [1]*remainding_slices
        chunk_offsets = [chunk_size*number_of_processes]*remainding_slices
        kernels = [kernel]*remainding_slices
        with ProcessPoolExecutor(max_workers=remainding_slices) as executor:
            for _ in executor.map(no_OF_filter_along_X_chunk,
                                  chunk_indexes,
                                  chunk_sizes,
                                  chunk_offsets,
                                  kernels):
                logging.debug(f"PU #{_} finished")

    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"Filtering along X spent {time_1 - time_0} seconds")

def no_OF_filter(kernels):
    no_OF_filter_along_Z(kernels[0])
    vol[...] = filtered_vol[...]
    no_OF_filter_along_Y(kernels[1])
    vol[...] = filtered_vol[...]
    no_OF_filter_along_X(kernels[2])

def int_or_str(text):
    '''Helper function for argument parsing.'''
    try:
        return int(text)
    except ValueError:
        return text

def feedback():
    while True:
        logging.info(f"{100*__percent__.value/np.sum(vol.shape):3.2f} % completed")
        time.sleep(1)

number_of_PUs = multiprocessing.cpu_count()

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
parser.add_argument("-s", "--sigma", nargs="+",
                    help="Gaussian sigma for each dimension in the order (Z, Y, X)",
                    default=(SIGMA, SIGMA, SIGMA))
parser.add_argument("-l", "--levels", type=int_or_str,
                    help="Number of levels of the Gaussian pyramid used by the optical flow estimator",
                    default=OF_LEVELS)
parser.add_argument("-w", "--winsize", type=int_or_str,
                    help="Size of the window used by the optical flow estimator",
                    default=OF_WINDOW_SIZE)
parser.add_argument("-v", "--verbosity", type=int_or_str,
                    help="Verbosity level", default=0)
parser.add_argument("-n", "--no_OF", action="store_true", help="Disable optical flow compensation")
parser.add_argument("-m", "--memory_map",
                    action="store_true",
                    help="Enable memory-mapping (see https://mrcfile.readthedocs.io/en/stable/usage_guide.html#dealing-with-large-files, only for MRC files)")
parser.add_argument("-p", "--number_of_processes", type=int_or_str,
                    help="Maximum number of processes",
                    default=number_of_PUs)
#parser.add_argument("--recompute_flow", action="store_true", help="Disable the use of adjacent optical flow fields")

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

    #number_of_detected_GPUs = cv2.cuda.getCudaEnabledDeviceCount()
    #if number_of_detected_GPUs > 0:
    #    logging.info(f"detected {number_of_detected_GPUs} GPUs")
    #    cv2.cuda.printShortCudaDeviceInfo(device=0)

    #if args.recompute_flow:
    #    get_flow = get_flow_without_prev_flow
    #    logging.info("No reusing adjacent OF fields as predictions")
    #else:
    #    get_flow = get_flow_with_prev_flow
    #    logging.info("Using adjacent OF fields as predictions")

    logging.info(f"Number of processing units: {number_of_PUs}")
        
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
            vol_MRC = mrcfile.open(args.input, mode="r+")
        vol = vol_MRC.data
    else:
        vol = skimage.io.imread(args.input, plugin="tifffile").astype(np.float32)
    vol_size = vol.dtype.itemsize * vol.size
    logging.info(f"shape of the input volume (Z, Y, X) = {vol.shape}")
    logging.info(f"type of the volume = {vol.dtype}")
    logging.info(f"vol requires {vol_size/(1024*1024):.1f} MB")
    logging.info(f"{args.input} max = {vol.max()}")
    logging.info(f"{args.input} min = {vol.min()}")
    vol_mean = vol.mean()
    logging.info(f"Input vol average = {vol_mean}")

    if __debug__:
        time_1 = time.perf_counter()
        logging.info(f"read \"{args.input}\" in {time_1 - time_0} seconds")

    kernels = [None]*3
    kernels[0] = get_gaussian_kernel(sigma[0])
    kernels[1] = get_gaussian_kernel(sigma[1])
    kernels[2] = get_gaussian_kernel(sigma[2])
    logging.info(f"length of each filter (Z, Y, X) = {[len(i) for i in [*kernels]]}")

    # Copy the volume to shared memory
    SM_vol = shared_memory.SharedMemory(
        create=True,
        size=vol_size,
        name="vol") # See /dev/shm/
    _vol = np.ndarray(
        shape=vol.shape,
        dtype=vol.dtype,
        buffer=SM_vol.buf)
    _vol[...] = vol[...]
    vol = _vol

    SM_filtered_vol = shared_memory.SharedMemory(
        create=True,
        size=vol_size,
        name="filtered_vol") # See /dev/shm
    filtered_vol = np.ndarray(
        shape=vol.shape,
        dtype=vol.dtype,
        buffer=SM_filtered_vol.buf)
    filtered_vol.fill(0)
    
    #vol = np.transpose(vol, transpose_pattern)
    #logging.info(f"After transposing, shape of the volume to denoise (Z, Y, X) = {vol.shape}")

    logging.info(f"Number of available processing units: {number_of_PUs}")
    number_of_processes = args.number_of_processes
    logging.info(f"Number of concurrent processes: {number_of_processes}")
    
    thread = threading.Thread(target=feedback)
    thread.daemon = True # To obey CTRL+C interruption.
    thread.start()

    if __debug__:
        logging.info(f"Filtering ...")
        #time_0 = time.perf_counter()
        time_0 = time.perf_counter()

    if args.no_OF:
        no_OF_filter(kernels)
    else:
        OF_filter(kernels, l, w)

    if __debug__:
        #time_1 = time.perf_counter()        
        time_1 = time.perf_counter()        
        logging.info(f"Volume filtered in {time_1 - time_0} seconds")

    #filtered_vol = np.transpose(filtered_vol, transpose_pattern)
    logging.info(f"{args.output} type = {filtered_vol.dtype}")
    logging.info(f"{args.output} max = {filtered_vol.max()}")
    logging.info(f"{args.output} min = {filtered_vol.min()}")
    logging.info(f"{args.output} average = {filtered_vol.mean()}")
    
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
        logging.debug(f"Writting TIFF file")
        skimage.io.imsave(args.output, filtered_vol.astype(np.float32), plugin="tifffile")

    SM_vol.close()
    SM_vol.unlink()
    SM_filtered_vol.close()
    SM_filtered_vol.unlink()
    
    if __debug__:
        time_1 = time.perf_counter()        
        logging.info(f"written \"{args.output}\" in {time_1 - time_0} seconds")
