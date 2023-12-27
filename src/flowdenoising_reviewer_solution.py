#!/usr/bin/env python
'''3D Gaussian filtering controlled by the optical flow.
'''

# "flowdenoising.py" is part of "https://github.com/microscopy-processing/FlowDenoising",
# authored by:
#
# * J.J. Fernández (CSIC).
# * V. González-Ruiz (UAL).
#
# Please, refer to the file LICENSE.txt to know the terms of usage of this software.

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
from multiprocessing import Value#, Lock
from multiprocessing.shared_memory import SharedMemory
#from multiprocessing import Process as Task
#from threading import Thread as Task
#from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from concurrent.futures.process import ProcessPoolExecutor as PoolExecutor

done = False

#def init_pool_processes(the_lock):
#    '''Initialize each process with a global variable lock.
#    '''
#    global lock
#    lock = the_lock

LOGGING_FORMAT = "[%(asctime)s] (%(levelname)s) %(message)s"

if __debug__:
    # In shared memory
    progress = Value('f', 0)
    OFE_time = Value('f', 0)
    warping_time = Value('f', 0)
    convolution_time = Value('f', 0)
    transference_time = Value('f', 0)

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
    logging.info(f"Kernel computed: {coeffs[1:-1]}")
    return coeffs[1:-1]

OFCA_EXTENSION_MODE = cv2.BORDER_REPLICATE
OF_LEVELS = 3
OF_WINDOW_SIZE = 5
OF_ITERS = 3
OF_POLY_N = 5
OF_POLY_SIGMA = 1.2
SIGMA = 2.0

def warp_slice(reference, flow):
    if __debug__:
        time_0 = time.perf_counter()
    height, width = flow.shape[:2]
    map_x = np.tile(np.arange(width), (height, 1))
    map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
    map_xy = (flow + np.dstack((map_x, map_y))).astype('float32')
    warped_slice = cv2.remap(reference, map_xy, None,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=OFCA_EXTENSION_MODE)
    if __debug__:
        time_1 = time.perf_counter()
        diff = time_1 - time_0
        warping_time.value += diff
    return warped_slice

def get_flow_with_prev_flow_CPU(
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
        diff = time_1 - time_0
        logging.debug(f"OF computed in \
{1000*(diff):4.3f} ms, max_X={np.max(flow[0]):+3.2f}, \
min_X={np.min(flow[0]):+3.2f}, max_Y={np.max(flow[1]):+3.2f}, \
min_Y={np.min(flow[1]):+3.2f}")
        OFE_time.value += diff
    return flow

def get_flow_without_prev_flow_CPU(
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
        diff = time_1 - time_0
        logging.debug(f"OF computed in {1000*(diff):4.3f} ms, \
max_X={np.max(flow[0]):+3.2f}, min_X={np.min(flow[0]):+3.2f}, \
max_Y={np.max(flow[1]):+3.2f}, min_Y={np.min(flow[1]):+3.2f}")
        OFE_time.value += diff
    return flow

def get_flow_with_prev_flow_GPU(GPU_reference, GPU_target, l=OF_LEVELS, w=OF_WINDOW_SIZE, GPU_prev_flow=None):
    if __debug__:
        time_0 = time.perf_counter()
    #GPU_target = cv2.cuda_GpuMat()
    #GPU_target.upload(target)
    #GPU_reference = cv2.cuda_GpuMat()
    #GPU_reference.upload(reference)
    #GPU_prev_flow = cv2.cuda_GpuMat()
    #GPU_prev_flow.upload(prev_flow)
    if __debug__:
        transference_time.value += (time.perf_counter() - time_0)

    if __debug__:
        time_0 = time.perf_counter()
    # create optical flow instance
    flower = cv2.cuda_FarnebackOpticalFlow.create(numLevels=l, pyrScale=0.5, fastPyramids=False, winSize=w, numIters=OF_ITERS, polyN=OF_POLY_N, polySigma=OF_POLY_SIGMA, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    #flower = cv2.cuda_FarnebackOpticalFlow.create(numLevels=l, pyrScale=0.5, fastPyramids=False, winSize=w, numIters=OF_ITERS, polyN=OF_POLY_N, polySigma=OF_POLY_SIGMA, flags=0)
    
    # calculate optical flow
    #gpu_flow = cv2.cuda.FarnebackOpticalFlow.calc(flower, I0=gpu_target, I1=gpu_reference, flow=None)
    GPU_flow = cv2.cuda.FarnebackOpticalFlow.calc(flower, I0=GPU_target, I1=GPU_reference, flow=GPU_prev_flow)
    if __debug__:
        time_1 = time.perf_counter()
        _OFE_time = time_1 - time_0
        OFE_time.value += _OFE_time

    if __debug__:
        time_2 = time.perf_counter()
    flow = GPU_flow.download()
    if __debug__:
        logging.debug(f"OF computed in {1000*(_OFE_time):4.3f} ms, max_X={np.max(flow[0]):+3.2f}, min_X={np.min(flow[0]):+3.2f}, max_Y={np.max(flow[1]):+3.2f}, min_Y={np.min(flow[1]):+3.2f}")
        transference_time.value += (time.perf_counter() - time_2)
    
    return flow

def get_flow_without_prev_flow_GPU(GPU_reference, GPU_target, l=OF_LEVELS, w=OF_WINDOW_SIZE, GPU_prev_flow=None):
    if __debug__:
        time_0 = time.perf_counter()
    #GPU_target = cv2.cuda_GpuMat()
    #GPU_target.upload(target)
    #GPU_reference = cv2.cuda_GpuMat()
    #GPU_reference.upload(reference)
    #GPU_prev_flow = cv2.cuda_GpuMat()
    #GPU_prev_flow.upload(prev_flow)
    if __debug__:
        transference_time.value += (time.perf_counter() - time_0)

    if __debug__:
        time_0 = time.perf_counter()
    # create optical flow instance
    flower = cv2.cuda_FarnebackOpticalFlow.create(numLevels=l, pyrScale=0.5, fastPyramids=False, winSize=w, numIters=OF_ITERS, polyN=OF_POLY_N, polySigma=OF_POLY_SIGMA, flags=0)
    #flower = cv2.cuda_FarnebackOpticalFlow.create(numLevels=l, pyrScale=0.5, fastPyramids=False, winSize=w, numIters=OF_ITERS, polyN=OF_POLY_N, polySigma=OF_POLY_SIGMA, flags=0)
    
    # calculate optical flow
    #gpu_flow = cv2.cuda.FarnebackOpticalFlow.calc(flower, I0=gpu_target, I1=gpu_reference, flow=None)
    GPU_flow = cv2.cuda.FarnebackOpticalFlow.calc(flower, I0=GPU_target, I1=GPU_reference, flow=None)
    print("hola")
    if __debug__:
        time_1 = time.perf_counter()
        _OFE_time = time_1 - time_0
        OFE_time.value += _OFE_time

    if __debug__:
        time_2 = time.perf_counter()
    flow = GPU_flow.download()
    if __debug__:
        logging.debug(f"OF computed in {1000*(_OFE_time):4.3f} ms, max_X={np.max(flow[0]):+3.2f}, min_X={np.min(flow[0]):+3.2f}, max_Y={np.max(flow[1]):+3.2f}, min_Y={np.min(flow[1]):+3.2f}")
        transference_time.value += (time.perf_counter() - time_2)
    
    return flow

class GaussianDenoising():

    def __init__(self, number_of_processes, kernels):
        #if __debug__:
        #    self.progress = Value('f', 0)
        self.number_of_processes = number_of_processes
        self.kernels = kernels

    def filter_along_Z_slice(self, z, kernel):
        ks2 = kernel.size//2
        tmp_slice = np.zeros(shape=(self.vol.shape[1], self.vol.shape[2]), dtype=np.float32)
        for i in range(kernel.size):
            tmp_slice += self.vol[(z + i - ks2) % self.vol.shape[0], :, :]*kernel[i]
        self.filtered_vol[z, :, :] = tmp_slice
        if __debug__:
            #self.progress.value += 1
            progress.value += 1
        #print("3", np.max(self.filtered_vol), self.filtered_vol.data)

    def filter_along_Y_slice(self, y, kernel):
        ks2 = kernel.size//2
        tmp_slice = np.zeros(shape=(self.vol.shape[0], self.vol.shape[2]), dtype=np.float32)
        for i in range(kernel.size):
            tmp_slice += self.vol[:, (y + i - ks2) % self.vol.shape[1], :]*kernel[i]
        self.filtered_vol[:, y, :] = tmp_slice
        if __debug__:
            #self.progress.value += 1
            progress.value += 1

    def filter_along_X_slice(self, x, kernel):
        ks2 = kernel.size//2
        tmp_slice = np.zeros(shape=(self.vol.shape[0], self.vol.shape[1]), dtype=np.float32)
        for i in range(kernel.size):
            tmp_slice += self.vol[:, :, (x + i - ks2) % self.vol.shape[2]]*kernel[i]
        self.filtered_vol[:, :, x] = tmp_slice
        if __debug__:
            #self.progress.value += 1
            progress.value += 1

    def filter_along_Z_chunk(self, chunk_index, chunk_size, chunk_offset, kernel):
        for z in range(chunk_size):
            self.filter_along_Z_slice(chunk_index*chunk_size + z + chunk_offset, kernel)
        return chunk_index # Probar a quitar este return
        #print("5", np.max(self.filtered_vol), self.filtered_vol.data)

    def filter_along_Y_chunk(self, chunk_index, chunk_size, chunk_offset, kernel):
        for y in range(chunk_size):
            self.filter_along_Y_slice(chunk_index*chunk_size + y + chunk_offset, kernel)
        return chunk_index

    def filter_along_X_chunk(self, chunk_index, chunk_size, chunk_offset, kernel):
        for x in range(chunk_size):
            self.filter_along_X_slice(chunk_index*chunk_size + x + chunk_offset, kernel)
        return chunk_index

    def filter_along_Z(self):
        kernel = self.kernels[0]
        logging.info(f"Filtering along Z")
        if __debug__:
            time_0 = time.perf_counter()
            min_OF = 1000
            max_OF = -1000
        Z_dim = self.vol.shape[0]
        chunk_size = Z_dim//self.number_of_processes
        #chunk_indexes = [i for i in range(self.number_of_processes)]
        #chunk_sizes = [chunk_size]*self.number_of_processes
        #chunk_offsets = [0]*self.number_of_processes
        #kernels = [kernel]*self.number_of_processes
        #with PoolExecutor(max_workers=self.number_of_processes) as executor:
        #    #print("4", np.max(self.filtered_vol), self.filtered_vol.data)
        #    for _ in executor.map(self.filter_along_Z_chunk,
        #                          chunk_indexes,
        #                          chunk_sizes,
        #                          chunk_offsets,
        #                          kernels):
        #        logging.debug(f"PU #{_} finished")
        #        #print("4", np.max(self.filtered_vol), self.filtered_vol.data)
        processes = []
        for i in range(self.number_of_processes):
            process = Task(
                target=self.filter_along_Z_chunk,
                args=(i, chunk_size, 0, kernel))
            process.start()
            processes.append(process)
        for i in processes:
            i.join()
                
        N_remaining_slices = Z_dim % self.number_of_processes
        if N_remaining_slices > 0:
            logging.info(f"remaining_slices={N_remaining_slices}")
            #chunk_indexes = [i for i in range(N_remaining_slices)]
            #chunk_sizes = [1]*N_remaining_slices
            #chunk_offsets = [chunk_size*self.number_of_processes]*N_remaining_slices
            #kernels = [kernel]*N_remaining_slices
            #with PoolExecutor(max_workers=N_remaining_slices) as executor:
            #    for _ in executor.map(self.filter_along_Z_chunk,
            #                          chunk_indexes,
            #                          chunk_sizes,
            #                          chunk_offsets,
            #                          kernels):
            #        logging.debug(f"PU #{_} finished")
            processes = []
            for i in range(N_remaining_slices):
                process = Task(
                    target=self.filter_along_Z_chunk,
                    args=(i, 1, chunk_size*self.number_of_processes, kernel))
                process.start()
                processes.append(process)
            for i in processes:
                i.join()

        if __debug__:
            time_1 = time.perf_counter()
            diff = time_1 - time_0
            logging.debug(f"Filtering along Z spent {diff} seconds")
            logging.debug(f"Min OF val: {min_OF}")
            logging.debug(f"Max OF val: {max_OF}")
            convolution_time.value += diff

    def filter_along_Y(self):
        kernel = self.kernels[1]
        logging.info(f"Filtering along Y")
        if __debug__:
            time_0 = time.perf_counter()
            min_OF = 1000
            max_OF = -1000
        
        Y_dim = self.vol.shape[1]
        chunk_size = Y_dim//self.number_of_processes
        #chunk_indexes = [i for i in range(self.number_of_processes)]
        #chunk_sizes = [chunk_size]*self.number_of_processes
        #chunk_offsets = [0]*self.number_of_processes
        #kernels = [kernel]*self.number_of_processes
        #with PoolExecutor(max_workers=self.number_of_processes) as executor:
        #    for _ in executor.map(self.filter_along_Y_chunk,
        #                          chunk_indexes,
        #                          chunk_sizes,
        #                          chunk_offsets,
        #                          kernels):
        #        logging.debug(f"PU #{_} finished")
        processes = []
        for i in range(self.number_of_processes):
            process = Task(
                target=self.filter_along_Y_chunk,
                args=(i, chunk_size, 0, kernel))
            process.start()
            processes.append(process)
        for i in processes:
            i.join()

        N_remaining_slices = Y_dim % self.number_of_processes
        if N_remaining_slices > 0:
            logging.info(f"remaining_slices={N_remaining_slices}")
            #chunk_indexes = [i for i in range(N_remaining_slices)]
            #chunk_sizes = [1]*N_remaining_slices
            #chunk_offsets = [chunk_size*self.number_of_processes]*N_remaining_slices
            #kernels = [kernel]*N_remaining_slices
            #with PoolExecutor(max_workers=N_remaining_slices) as executor:
            #    for _ in executor.map(self.filter_along_Y_chunk,
            #                          chunk_indexes,
            #                          chunk_sizes,
            #                          chunk_offsets,
            #                          kernels):
            #        logging.debug(f"PU #{_} finished")
            processes = []
            for i in range(N_remaining_slices):
                process = Task(
                    target=self.filter_along_Y_chunk,
                    args=(i, 1, chunk_size*self.number_of_processes, kernel))
                process.start()
                processes.append(process)
            for i in processes:
                i.join()

        if __debug__:
            time_1 = time.perf_counter()
            diff = time_1 - time_0
            logging.debug(f"Filtering along Y spent {diff} seconds")
            logging.debug(f"Min OF val: {min_OF}")
            logging.debug(f"Max OF val: {max_OF}")
            convolution_time.value += diff

    def filter_along_X(self):
        kernel = self.kernels[2]
        logging.info(f"Filtering along X")
        if __debug__:
            time_0 = time.perf_counter()
            min_OF = 1000
            max_OF = -1000
        
        X_dim = vol.shape[2]
        chunk_size = X_dim//self.number_of_processes
        #chunk_indexes = [i for i in range(self.number_of_processes)]
        #chunk_sizes = [chunk_size]*self.number_of_processes
        #chunk_offsets = [0]*self.number_of_processes
        #kernels = [kernel]*self.number_of_processes
        #with PoolExecutor(max_workers=self.number_of_processes) as executor:
        #    for _ in executor.map(self.filter_along_X_chunk,
        #                          chunk_indexes,
        #                          chunk_sizes,
        #                          chunk_offsets,
        #                          kernels):
        #        logging.debug(f"PU #{_} finished")
        processes = []
        for i in range(self.number_of_processes):
            process = Task(
                target=self.filter_along_X_chunk,
                args=(i, chunk_size, 0, kernel))
            process.start()
            processes.append(process)
        for i in processes:
            i.join()        
        N_remaining_slices = X_dim % self.number_of_processes
        if N_remaining_slices > 0:
            logging.info(f"remaining_slices={N_remaining_slices}")
            #chunk_indexes = [i for i in range(N_remaining_slices)]
            #chunk_sizes = [1]*N_remaining_slices
            #chunk_offsets = [chunk_size*self.number_of_processes]*N_remaining_slices
            #kernels = [kernel]*N_remaining_slices
            #with PoolExecutor(max_workers=N_remaining_slices) as executor:
            #    for _ in executor.map(self.filter_along_X_chunk,
            #                          chunk_indexes,
            #                          chunk_sizes,
            #                          chunk_offsets,
            #                          kernels):
            #        logging.debug(f"PU #{_} finished")
            processes = []
            for i in range(N_remaining_slices):
                process = Task(
                    target=self.filter_along_X_chunk,
                    args=(i, 1, chunk_size*self.number_of_processes, kernel))
                process.start()
                processes.append(process)
            for i in processes:
                i.join()

        if __debug__:
            time_1 = time.perf_counter()
            diff = time_1 - time_0
            logging.debug(f"Filtering along X spent {diff} seconds")
            logging.debug(f"Min OF val: {min_OF}")
            logging.debug(f"Max OF val: {max_OF}")
            convolution_time.value += diff
        
    def filter(self, vol):
        vol_size = vol.dtype.itemsize*vol.size
        #self.vol = vol # This only can done if we were using threads
        self.SM_vol = SharedMemory(
            create=True,
            size=vol_size,
            name="vol") # See /dev/shm/vol
        self.vol = np.ndarray(
            shape=vol.shape,
            dtype=vol.dtype,
            buffer=self.SM_vol.buf)
        self.vol = vol.copy()
        #if __debug__:
        #    logging.info(f"shape of the input volume (Z, Y, X) = {self.vol.shape}")
        #    logging.info(f"type of the volume = {self.vol.dtype}")
        #    logging.info(f"vol requires {vol_size/(1024*1024):.1f} MB")
        #    logging.info(f"{args.input} max = {self.vol.max()}")
        #    logging.info(f"{args.input} min = {self.vol.min()}")
        #    vol_mean = vol.mean()
        #    logging.info(f"Input vol average = {vol_mean}")
        #self.filtered_vol = np.zeros_like(vol) # This only can done if we were using threads
        self.SM_filtered_vol = SharedMemory(
            create=True,
            size=vol_size,
            name="filtered_vol") # See /dev/shm/filtered_vol
        self.filtered_vol = np.ndarray(
            shape=vol.shape,
            dtype=vol.dtype,
            buffer=self.SM_filtered_vol.buf)
        self.filtered_vol.fill(0)
        #print("1", np.max(self.filtered_vol), self.filtered_vol.data)
        self.filter_along_Z()
        #print("2", np.max(self.filtered_vol), self.filtered_vol.data)
        #self.vol[...] = self.filtered_vol[...]
        #self.filter_along_Y()
        #self.vol[...] = self.filtered_vol[...]
        #self.filter_along_X()
        return self.filtered_vol
        #return self.vol

    def close(self):
        self.SM_vol.close()
        self.SM_vol.unlink()
        self.SM_filtered_vol.close()
        self.SM_filtered_vol.unlink()

    def feedback(self):
        global done
        while not done:
            #logging.info(f"{100*self.progress.value/np.sum(vol.shape):3.2f} % filtering completed")
            logging.info(f"{100*progress.value/np.sum(vol.shape):3.2f} % filtering completed")
            time.sleep(1)

class FlowDenoising(GaussianDenoising):

    def __init__(self, number_of_processes, kernels, l, w, get_flow, warp_slice):
        super().__init__(number_of_processes, kernels)
        self.l = l
        self.w = w
        self.get_flow = get_flow
        self.warp_slice = warp_slice

    def filter(self, vol):
        #reshaped_vol = vol.reshape(-1, vol.shape[0])
        #self.GPU_vol = cv2.cuda_GpuMat()
        #self.GPU_vol.create(reshaped_vol.shape[0], reshaped_vol.shape[1], cv2.CV_32F)
        #self.GPU_vol.upload(reshaped_vol)

        self.GPU_vol = []
        for z in range(vol.shape[0]):
            self.GPU_vol.append(cv2.cuda_GpuMat())
            print(".", end='')
        for z in range(vol.shape[0]): # Se puede quitar, creo
            print("o", end='')
            self.GPU_vol[z].create(vol.shape[1], vol.shape[2], cv2.CV_32F)
        for z in range(vol.shape[0]):
            print("O", end='')
            self.GPU_vol[z].upload(vol[z])
        
        #input()
        self.filtered_vol = super().filter(vol)
        return self.filtered_vol

    def filter_along_Z_slice(self, z, kernel):
        print("z=", z)
        ks2 = kernel.size//2
        tmp_slice = np.zeros_like(self.vol[z, :, :]).astype(np.float32)
        assert kernel.size % 2 != 0 # kernel.size must be odd
        prev_flow = np.zeros(shape=(self.vol.shape[1], self.vol.shape[2], 2), dtype=np.float32)
        for i in range(ks2 - 1, -1, -1):
            flow = self.get_flow(self.GPU_vol[(z + i - ks2) % self.vol.shape[0]],
                                 self.GPU_vol[z], l, w, None)
            prev_flow = flow
            OF_compensated_slice = self.warp_slice(self.vol[(z + i - ks2) % self.vol.shape[0], :, :], flow)
            tmp_slice += OF_compensated_slice*kernel[i]
            print(i)
        tmp_slice += self.vol[z, :, :]*kernel[ks2]
        prev_flow = np.zeros(shape=(self.vol.shape[1], self.vol.shape[2], 2), dtype=np.float32)
        for i in range(ks2 + 1, kernel.size):
            flow = self.get_flow(self.GPU_vol[(z + i - ks2) % self.vol.shape[0]],
                                 self.GPU_vol[z], l, w, None)
            prev_flow = flow
            OF_compensated_slice = self.warp_slice(self.vol[(z + i - ks2) % self.vol.shape[0], :, :], flow)
            tmp_slice += OF_compensated_slice*kernel[i]
        self.filtered_vol[z, :, :] = tmp_slice
        if __debug__:
            #self.progress.value += 1
            progress.value += 1

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
            tmp_slice += OF_compensated_slice*kernel[i]
        tmp_slice += self.vol[:, y, :]*kernel[ks2]
        prev_flow = np.zeros(shape=(self.vol.shape[0], self.vol.shape[2], 2), dtype=np.float32)
        for i in range(ks2 + 1, kernel.size):
            flow = self.get_flow(self.vol[:, (y + i - ks2) % self.vol.shape[1], :],
                                 self.vol[:, y, :], l, w, prev_flow)
            prev_flow = flow
            OF_compensated_slice = self.warp_slice(self.vol[:, (y + i - ks2) % self.vol.shape[1], :], flow)
            tmp_slice += OF_compensated_slice*kernel[i]
        self.filtered_vol[:, y, :] = tmp_slice
        if __debug__:
            #self.progress.value += 1
            progress.value += 1

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
            tmp_slice += OF_compensated_slice*kernel[i]
        tmp_slice += self.vol[:, :, x]*kernel[ks2]
        prev_flow = np.zeros(shape=(self.vol.shape[0], self.vol.shape[1], 2), dtype=np.float32)
        for i in range(ks2 + 1, kernel.size):
            flow = get_flow(self.vol[:, :, (x + i - ks2) % self.vol.shape[2]],
                            self.vol[:, :, x], l, w, prev_flow)
            prev_flow = flow
            OF_compensated_slice = self.warp_slice(self.vol[:, :, (x + i - ks2) % self.vol.shape[2]], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        self.filtered_vol[:, :, x] = tmp_slice
        if __debug__:
            #self.progress.value += 1
            progress.value += 1

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
parser.add_argument("--use_GPU", action="store_true", help="Compute the optical flow in the GPU (if available through CUDA)")
parser.add_argument("--use_threads", action="store_true", help="Use threads instead of processes")

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

    if args.use_threads:
        from threading import Thread as Task
        logging.info("Using threads")
    else:
        from multiprocessing import Process as Task
        logging.info("Using processes")

    if args.recompute_flow:
        if args.use_GPU:
            get_flow = get_flow_without_prev_flow_GPU
            logging.info("Computing the optical flow in the GPU")
        else:
            get_flow = get_flow_without_prev_flow_CPU
            logging.info("Computing the optical flow in the CPU")
        logging.info("No reusing adjacent OF fields as predictions")
    else:
        if args.use_GPU:
            get_flow = get_flow_with_prev_flow_GPU
            logging.info("Computing the optical flow in the GPU")
        else:
            get_flow = get_flow_with_prev_flow_CPU
            logging.info("Computing the optical flow in the CPU")
        logging.info("Using adjacent OF fields as predictions")

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
    vol_size = vol.dtype.itemsize * vol.size

    logging.info(f"shape of the input volume (Z, Y, X) = {vol.shape}")
    logging.info(f"type of the volume = {vol.dtype}")
    logging.info(f"vol requires {vol_size/(1024*1024):.1f} MB")
    logging.info(f"{args.input} max = {vol.max()}")
    logging.info(f"{args.input} min = {vol.min()}")
    if __debug__:
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
    
    #vol = np.transpose(vol, transpose_pattern)
    #logging.info(f"After transposing, shape of the volume to denoise (Z, Y, X) = {vol.shape}")

    number_of_processes = args.number_of_processes
    logging.info(f"Number of available processing units: {number_of_PUs}")
    logging.info(f"Number of concurrent processes: {number_of_processes}")

    logging.info(f"Filtering ...")
    if __debug__:
        time_0 = time.perf_counter()

    if args.no_OF:
        fd = GaussianDenoising(number_of_processes, kernels)
    else:
        fd = FlowDenoising(number_of_processes, kernels, l, w, get_flow, warp_slice)

    if __debug__:
        thread = threading.Thread(target=fd.feedback)
        thread.daemon = True # To obey CTRL+C interruption.
        thread.start()

    filtered_vol = fd.filter(vol)

    #logging.info(f"{args.input} type = {vol.dtype}")
    #logging.info(f"{args.input} max = {vol.max()}")
    #logging.info(f"{args.input} min = {vol.min()}")
    #logging.info(f"{args.input} average = {vol.mean()}")
    if __debug__:
        time_1 = time.perf_counter()        
        logging.info(f"Volume filtered in {time_1 - time_0} seconds")

    #print(type(filtered_vol), filtered_vol.shape, filtered_vol.dtype)
    #quit()
    #print(np.max(filtered_vol))

    #filtered_vol = np.transpose(filtered_vol, transpose_pattern)
    logging.info(f"{args.output} type = {filtered_vol.dtype}")
    logging.info(f"{args.output} max = {filtered_vol.max()}")
    logging.info(f"{args.output} min = {filtered_vol.min()}")
    logging.info(f"{args.output} average = {filtered_vol.mean()}")
    
    if __debug__:
        logging.info(f"writing \"{args.output}\"")
        time_0 = time.perf_counter()
        logging.debug(f"output = {args.output}")

    #MRC_output = "mrc" in args.output.split('.')[-1].lower()
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
        logging.info(f"OFE_time = {OFE_time.value/number_of_processes} seconds")
        logging.info(f"warping_time = {warping_time.value/number_of_processes} seconds")
        logging.info(f"convolution_time = {convolution_time.value/number_of_processes} seconds")
        logging.info(f"transference_time = {transference_time.value} seconds")

    fd.close()
    done = True
    print("done")
