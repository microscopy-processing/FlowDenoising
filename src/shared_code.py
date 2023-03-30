# "shared_code.py" is part of "https://github.com/microscopy-processing/FlowDenoising", authored by:
#
# * J.J. Fernández (CSIC).
# * V. González-Ruiz (UAL).
#
# This module contains the shared code among the different implementations of FlowDenoising.
#
# Please, refer to the LICENSE.txt to know the terms of usage of this software.

import cv2
import numpy as np
import scipy.ndimage
import logging

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

