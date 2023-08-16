#!/usr/bin/env python
'''3D Gaussian filtering controlled by the optical flow.
'''

# "flowdenoising.py" is part of "https://github.com/microscopy-processing/FlowDenoising", authored by:
#
# * J.J. Fernández (CSIC).
# * V. González-Ruiz (UAL).
#
# Please, refer to the LICENSE.txt to know the terms of usage of this software.

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
import sys
import hashlib
import concurrent
import multiprocessing
from concurrent.futures import ThreadPoolExecutor as PoolExecutor

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
    warped_slice = cv2.remap(reference, map_xy, None,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=OFCA_EXTENSION_MODE)
    return warped_slice

def get_flow_with_prev_flow(
        reference, target, l=OF_LEVELS, w=OF_WINDOW_SIZE, prev_flow=None):
    if __debug__:
        time_0 = time.perf_counter()
    flow = cv2.calcOpticalFlowFarneback(
        prev=target,
        next=reference,
        flow=prev_flow,
        pyr_scale=0.5,
        levels=l,
        winsize=w,
        iterations=OF_ITERS,
        poly_n=OF_POLY_N,
        poly_sigma=OF_POLY_SIGMA,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    #flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=None, pyr_scale=0.5, levels=l, winsize=w, iterations=OF_ITERS, poly_n=OF_POLY_N, poly_sigma=OF_POLY_SIGMA, flags=0)
    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"OF computed in \
{1000*(time_1 - time_0):4.3f} ms, max_X={np.max(flow[0]):+3.2f}, \
min_X={np.min(flow[0]):+3.2f}, max_Y={np.max(flow[1]):+3.2f}, \
min_Y={np.min(flow[1]):+3.2f}")
    return flow

def get_flow_without_prev_flow(
        reference,
        target,
        l=OF_LEVELS,
        w=OF_WINDOW_SIZE,
        prev_flow=None):
    if __debug__:
        time_0 = time.perf_counter()
    #flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=prev_flow, pyr_scale=0.5, levels=l, winsize=w, iterations=OF_ITERS, poly_n=OF_POLY_N, poly_sigma=OF_POLY_SIGMA, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    flow = cv2.calcOpticalFlowFarneback(
        prev=target,
        next=reference,
        flow=None,
        pyr_scale=0.5,
        levels=l,
        winsize=w,
        iterations=OF_ITERS,
        poly_n=OF_POLY_N,
        poly_sigma=OF_POLY_SIGMA,
        flags=0)
    if __debug__:
        time_1 = time.perf_counter()
        logging.debug(f"OF computed in {1000*(time_1 - time_0):4.3f} ms, \
max_X={np.max(flow[0]):+3.2f}, min_X={np.min(flow[0]):+3.2f}, \
max_Y={np.max(flow[1]):+3.2f}, min_Y={np.min(flow[1]):+3.2f}")
    return flow

class GaussianDenoising():

    def __init__(self, number_of_processes, vol):
        if __debug__:
            self.progress = 0.0
        self.number_of_processes = number_of_processes
        self.vol = vol
        vol_size = vol.dtype.itemsize * vol.size
        logging.info(f"shape of the input volume (Z, Y, X) = {vol.shape}")
        logging.info(f"type of the volume = {vol.dtype}")
        logging.info(f"vol requires {vol_size/(1024*1024):.1f} MB")
        logging.info(f"{args.input} max = {vol.max()}")
        logging.info(f"{args.input} min = {vol.min()}")
        vol_mean = vol.mean()
        logging.info(f"Input vol average = {vol_mean}")
        self.filtered_vol = np.zeros_like(vol)

    def filter_along_Z_slice(self, z, kernel):
        ks2 = kernel.size//2
        tmp_slice = np.zeros(shape=(self.vol.shape[1], self.vol.shape[2]), dtype=np.float32)
        for i in range(kernel.size):
            tmp_slice += self.vol[(z + i - ks2) % self.vol.shape[0], :, :]*kernel[i]
        self.filtered_vol[z, :, :] = tmp_slice
        if __debug__:
            self.progress += 1

    def filter_along_Y_slice(self, y, kernel):
        ks2 = kernel.size//2
        tmp_slice = np.zeros(shape=(self.vol.shape[0], self.vol.shape[2]), dtype=np.float32)
        for i in range(kernel.size):
            tmp_slice += self.vol[:, (y + i - ks2) % self.vol.shape[1], :]*kernel[i]
        self.filtered_vol[:, y, :] = tmp_slice
        if __debug__:
            self.progress += 1

    def filter_along_X_slice(self, x, kernel):
        ks2 = kernel.size//2
        tmp_slice = np.zeros(shape=(self.vol.shape[0], self.vol.shape[1]), dtype=np.float32)
        for i in range(kernel.size):
            tmp_slice += self.vol[:, :, (x + i - ks2) % self.vol.shape[2]]*kernel[i]
        self.filtered_vol[:, :, x] = tmp_slice
        if __debug__:
            self.progress += 1

    def filter_along_Z_chunk(self, chunk_index, chunk_size, chunk_offset, kernel):
        for z in range(chunk_size):
            self.filter_along_Z_slice(chunk_index*chunk_size + z + chunk_offset, kernel)
        return chunk_index

    def filter_along_Y_chunk(self, chunk_index, chunk_size, chunk_offset, kernel):
        for y in range(chunk_size):
            self.filter_along_Y_slice(chunk_index*chunk_size + y + chunk_offset, kernel)
        return chunk_index

    def filter_along_X_chunk(self, chunk_index, chunk_size, chunk_offset, kernel):
        for x in range(chunk_size):
            self.filter_along_X_slice(chunk_index*chunk_size + x + chunk_offset, kernel)
        return chunk_index

    def filter_along_Z(self, kernel):
        logging.info(f"Filtering along Z with kernel length={kernel.size}")

        if __debug__:
            time_0 = time.perf_counter()

        Z_dim = self.vol.shape[0]
        chunk_size = Z_dim//self.number_of_processes
        chunk_indexes = [i for i in range(self.number_of_processes)]
        chunk_sizes = [chunk_size]*self.number_of_processes
        chunk_offsets = [0]*self.number_of_processes
        kernels = [kernel]*self.number_of_processes
        with PoolExecutor(max_workers=self.number_of_processes) as executor:
            for _ in executor.map(self.filter_along_Z_chunk,
                                  chunk_indexes,
                                  chunk_sizes,
                                  chunk_offsets,
                                  kernels):
                logging.debug(f"PU #{_} finished")
        N_remaining_slices = Z_dim % self.number_of_processes
        if N_remaining_slices > 0:
            chunk_indexes = [i for i in range(N_remaining_slices)]
            chunk_sizes = [1]*N_remaining_slices
            chunk_offsets = [chunk_size*self.number_of_processes]*N_remaining_slices
            kernels = [kernel]*N_remaining_slices
            with PoolExecutor(max_workers=N_remaining_slices) as executor:
                for _ in executor.map(self.filter_along_Z_chunk,
                                      chunk_indexes,
                                      chunk_sizes,
                                      chunk_offsets,
                                      kernels):
                    logging.debug(f"PU #{_} finished")

        if __debug__:
            time_1 = time.perf_counter()
            logging.debug(f"Filtering along Z spent {time_1 - time_0} seconds")

    def filter_along_Y(self, kernel):
        logging.info(f"Filtering along Y with kernel length={kernel.size}")

        if __debug__:
            time_0 = time.perf_counter()

        Y_dim = self.vol.shape[1]
        chunk_size = Y_dim//self.number_of_processes
        chunk_indexes = [i for i in range(self.number_of_processes)]
        chunk_sizes = [chunk_size]*self.number_of_processes
        chunk_offsets = [0]*self.number_of_processes
        kernels = [kernel]*self.number_of_processes
        with PoolExecutor(max_workers=self.number_of_processes) as executor:
            for _ in executor.map(self.filter_along_Y_chunk,
                                  chunk_indexes,
                                  chunk_sizes,
                                  chunk_offsets,
                                  kernels):
                logging.debug(f"PU #{_} finished")
        N_remaining_slices = Y_dim % self.number_of_processes
        if N_remaining_slices > 0:
            chunk_indexes = [i for i in range(N_remaining_slices)]
            chunk_sizes = [1]*N_remaining_slices
            chunk_offsets = [chunk_size*self.number_of_processes]*N_remaining_slices
            kernels = [kernel]*N_remaining_slices
            with PoolExecutor(max_workers=N_remaining_slices) as executor:
                for _ in executor.map(self.filter_along_Y_chunk,
                                      chunk_indexes,
                                      chunk_sizes,
                                      chunk_offsets,
                                      kernels):
                    logging.debug(f"PU #{_} finished")

        if __debug__:
            time_1 = time.perf_counter()
            logging.debug(f"Filtering along Y spent {time_1 - time_0} seconds")

    def filter_along_X(self, kernel):
        logging.info(f"Filtering along X with kernel length={kernel.size}")
        if __debug__:
            time_0 = time.perf_counter()

        X_dim = vol.shape[2]
        chunk_size = X_dim//self.number_of_processes
        chunk_indexes = [i for i in range(self.number_of_processes)]
        chunk_sizes = [chunk_size]*self.number_of_processes
        chunk_offsets = [0]*self.number_of_processes
        kernels = [kernel]*self.number_of_processes
        with PoolExecutor(max_workers=self.number_of_processes) as executor:
            for _ in executor.map(self.filter_along_X_chunk,
                                  chunk_indexes,
                                  chunk_sizes,
                                  chunk_offsets,
                                  kernels):
                logging.debug(f"PU #{_} finished")
        N_remaining_slices = X_dim % self.number_of_processes
        if N_remaining_slices > 0:
            chunk_indexes = [i for i in range(N_remaining_slices)]
            chunk_sizes = [1]*N_remaining_slices
            chunk_offsets = [chunk_size*self.number_of_processes]*N_remaining_slices
            kernels = [kernel]*N_remaining_slices
            with PoolExecutor(max_workers=N_remaining_slices) as executor:
                for _ in executor.map(self.filter_along_X_chunk,
                                      chunk_indexes,
                                      chunk_sizes,
                                      chunk_offsets,
                                      kernels):
                    logging.debug(f"PU #{_} finished")

        if __debug__:
            time_1 = time.perf_counter()
            logging.debug(f"Filtering along X spent {time_1 - time_0} seconds")

    def filter(self, kernels):
        self.filter_along_Z(kernels[0])
        self.vol[...] = self.filtered_vol[...]
        self.filter_along_Y(kernels[1])
        self.vol[...] = self.filtered_vol[...]
        self.filter_along_X(kernels[2])

    def feedback(self):
        while True:
            logging.info(f"{100*self.progress/np.sum(vol.shape):3.2f} % filtering completed")
            time.sleep(1)

class FlowDenoising(GaussianDenoising):

    def __init__(self, number_of_processes, vol, l, w, get_flow, warp_slice):
        super().__init__(number_of_processes, vol)
        self.l = l
        self.w = w
        self.get_flow = get_flow
        self.warp_slice = warp_slice

    def filter_along_Z_slice(self, z, kernel):
        ks2 = kernel.size//2
        tmp_slice = np.zeros_like(self.vol[z, :, :]).astype(np.float32)
        assert kernel.size % 2 != 0 # kernel.size must be odd
        prev_flow = np.zeros(shape=(self.vol.shape[1], self.vol.shape[2], 2), dtype=np.float32)
        for i in range(ks2 - 1, -1, -1):
            flow = self.get_flow(self.vol[(z + i - ks2) % self.vol.shape[0], :, :],
                                 self.vol[z, :, :], l, w, prev_flow)
            prev_flow = flow
            OF_compensated_slice = self.warp_slice(self.vol[(z + i - ks2) % self.vol.shape[0], :, :], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        tmp_slice += self.vol[z, :, :] * kernel[ks2]
        prev_flow = np.zeros(shape=(self.vol.shape[1], self.vol.shape[2], 2), dtype=np.float32)
        for i in range(ks2 + 1, kernel.size):
            flow = self.get_flow(self.vol[(z + i - ks2) % self.vol.shape[0], :, :],
                                 self.vol[z, :, :], l, w, prev_flow)
            prev_flow = flow
            OF_compensated_slice = self.warp_slice(self.vol[(z + i - ks2) % self.vol.shape[0], :, :], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        self.filtered_vol[z, :, :] = tmp_slice
        if __debug__:
            self.progress += 1

    def filter_along_Y_slice(self, y, kernel):
        ks2 = kernel.size//2
        tmp_slice = np.zeros_like(self.vol[:, y, :]).astype(np.float32)
        assert kernel.size % 2 != 0 # kernel.size must be odd
        prev_flow = np.zeros(shape=(self.vol.shape[0], self.vol.shape[2], 2), dtype=np.float32)
        for i in range(ks2 - 1, -1, -1):
            flow = self.get_flow(self.vol[:, (y + i - ks2) % self.vol.shape[1], :],
                                 self.vol[:, y, :], l, w, prev_flow)
            prev_flow = flow
            OF_compensated_slice = self.warp_slice(self.vol[:, (y + i - ks2) % self.vol.shape[1], :], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        tmp_slice += self.vol[:, y, :] * kernel[ks2]
        prev_flow = np.zeros(shape=(self.vol.shape[0], self.vol.shape[2], 2), dtype=np.float32)
        for i in range(ks2 + 1, kernel.size):
            flow = self.get_flow(self.vol[:, (y + i - ks2) % self.vol.shape[1], :],
                                 self.vol[:, y, :], l, w, prev_flow)
            prev_flow = flow
            OF_compensated_slice = self.warp_slice(self.vol[:, (y + i - ks2) % self.vol.shape[1], :], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        self.filtered_vol[:, y, :] = tmp_slice
        if __debug__:
            self.progress += 1

    def filter_along_X_slice(self, x, kernel):
        ks2 = kernel.size//2
        tmp_slice = np.zeros_like(self.vol[:, :, x]).astype(np.float32)
        assert kernel.size % 2 != 0 # kernel.size must be odd
        prev_flow = np.zeros(shape=(self.vol.shape[0], self.vol.shape[1], 2), dtype=np.float32)
        for i in range(ks2 - 1, -1, -1):
            flow = self.get_flow(self.vol[:, :, (x + i - ks2) % self.vol.shape[2]],
                                 self.vol[:, :, x], l, w, prev_flow)
            prev_flow = flow
            OF_compensated_slice = self.warp_slice(self.vol[:, :, (x + i - ks2) % self.vol.shape[2]], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        tmp_slice += self.vol[:, :, x] * kernel[ks2]
        prev_flow = np.zeros(shape=(self.vol.shape[0], self.vol.shape[1], 2), dtype=np.float32)
        for i in range(ks2 + 1, kernel.size):
            flow = get_flow(self.vol[:, :, (x + i - ks2) % self.vol.shape[2]],
                            self.vol[:, :, x], l, w, prev_flow)
            prev_flow = flow
            OF_compensated_slice = self.warp_slice(self.vol[:, :, (x + i - ks2) % self.vol.shape[2]], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        self.filtered_vol[:, :, x] = tmp_slice
        if __debug__:
            self.progress += 1

def int_or_str(text):
    '''Helper function for argument parsing.'''
    try:
        return int(text)
    except ValueError:
        return text

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
parser.add_argument("--recompute_flow", action="store_true", help="Disable the use of adjacent optical flow fields")

parser.add_argument("--show_fingerprint", action="store_true", help="Show a hash of this file (you must run it in the folder that contains flowdenoising.py)")

def show_memory_usage(msg=''):
    logging.info(f"{psutil.Process(os.getpid()).memory_info().rss/(1024*1024):.1f} MB used in process {os.getpid()} {msg}")

if __name__ == "__main__":    
    parser.description = __doc__
    print ("Python version =", sys.version)
    
    args = parser.parse_args()
    if args.show_fingerprint:
        hash_algorithm = hashlib.new(name="sha256")
        with open("flowdenoising.py", "rb") as file:
            while chunk := file.read(512):
                hash_algorithm.update(chunk)
        fingerprint = hash_algorithm.hexdigest()
        print("fingerprint =", fingerprint)

    if args.verbosity == 2:
        logging.basicConfig(format=LOGGING_FORMAT, level=logging.DEBUG)
        logging.info("Verbosity level = 2")
    elif args.verbosity == 1:
        logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)
        logging.info("Verbosity level = 1")        
    else:
        logging.basicConfig(format=LOGGING_FORMAT, level=logging.CRITICAL)

    if args.recompute_flow:
        get_flow = get_flow_without_prev_flow
        logging.info("No reusing adjacent OF fields as predictions")
    else:
        get_flow = get_flow_with_prev_flow
        logging.info("Using adjacent OF fields as predictions")

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

    MRC_input = "mrc" in args.input.split('.')[-1].lower()
    if MRC_input:
        #if args.memory_map:
        #    logging.info(f"Using memory mapping")
        #    vol_MRC = rc = mrcfile.mmap(args.input, mode='r+')
        #else:
        with mrcfile.open(args.input, mode="r+") as vol_MRC:
            vol = vol_MRC.data
    else:
        vol = skimage.io.imread(args.input, plugin="tifffile").astype(np.float32)

    if __debug__:
        time_1 = time.perf_counter()
        logging.info(f"read \"{args.input}\" in {time_1 - time_0} seconds")

    kernels = [None]*3
    kernels[0] = get_gaussian_kernel(sigma[0])
    kernels[1] = get_gaussian_kernel(sigma[1])
    kernels[2] = get_gaussian_kernel(sigma[2])
    logging.info(f"length of each filter (Z, Y, X) = {[len(i) for i in [*kernels]]}")
    
    #vol = np.transpose(vol, transpose_pattern)
    #logging.info(f"After transposing, shape of the volume to denoise (Z, Y, X) = {vol.shape}")

    logging.info(f"Number of available processing units: {number_of_PUs}")
    number_of_processes = args.number_of_processes
    logging.info(f"Number of concurrent processes: {number_of_processes}")

    if __debug__:
        logging.info(f"Filtering ...")
        time_0 = time.perf_counter()

    logging.info(f"{args.input} type = {vol.dtype}")
    logging.info(f"{args.input} max = {vol.max()}")
    logging.info(f"{args.input} min = {vol.min()}")
    logging.info(f"{args.input} average = {vol.mean()}")

    if args.no_OF:
        fd = GaussianDenoising(number_of_processes, vol)
    else:
        fd = FlowDenoising(number_of_processes, vol, l, w, get_flow, warp_slice)

    if __debug__:
        thread = threading.Thread(target=fd.feedback)
        thread.daemon = True # To obey CTRL+C interruption.
        thread.start()
    
    filtered_vol = fd.filter(kernels)

    logging.info(f"{args.input} type = {vol.dtype}")
    logging.info(f"{args.input} max = {vol.max()}")
    logging.info(f"{args.input} min = {vol.min()}")
    logging.info(f"{args.input} average = {vol.mean()}")
    
    filtered_vol = vol.copy()

    if __debug__:
        time_1 = time.perf_counter()        
        logging.info(f"Volume filtered in {time_1 - time_0} seconds")

    #filtered_vol = np.transpose(filtered_vol, transpose_pattern)

    logging.info(f"{args.output} type = {filtered_vol.dtype}")
    logging.info(f"{args.output} max = {filtered_vol.max()}")
    logging.info(f"{args.output} min = {filtered_vol.min()}")
    logging.info(f"{args.output} average = {filtered_vol.mean()}")
    
    if __debug__:
        logging.info(f"writing \"{args.output}\"")
        time_0 = time.perf_counter()

    logging.debug(f"output = {args.output}")

    MRC_output = ( args.output.split('.')[-1] == "MRC" or args.output.split('.')[-1] == "mrc" )

    if MRC_output:
        logging.debug(f"Writing MRC file")
        with mrcfile.new(args.output, overwrite=True) as mrc:
            mrc.set_data(filtered_vol.astype(np.float32))
            mrc.data
    else:
        logging.debug(f"Writting TIFF file")
        skimage.io.imsave(args.output, filtered_vol.astype(np.float32), plugin="tifffile")
    
    if __debug__:
        time_1 = time.perf_counter()        
        logging.info(f"written \"{args.output}\" in {time_1 - time_0} seconds")
