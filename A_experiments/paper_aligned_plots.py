
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import sys

sys.path.append('..')
from src.compress_sensing import *
from src.utility import *
from A_experiments.theta_exp_improved import *

import numpy as np
from src.utility import *

# patch sizes aand n values from paper
#  d ∈ {8 × 8,16 × 16,32 × 32}
#  n ∈ {8,14,20,26,32}, {32,56,80,104,128}, and {128,224,320,416,512}
# patch_sampling = {
#     8: [8, 14, 20, 26, 32],
#     16: [32, 56, 80, 104, 128],
#     32: [128, 224, 320, 416, 512]
# }

PATCH_SIZE = 32
N_OBS = 256
ALPHA = 1
CELL_SIZE = 7
BLOB_SIZE = 2

# TODO: what are the V1 params s and f?

def extract_patches(img, patch_size):
    '''
        extract all patch_size x patch_size patches from img
    '''
    h, w = img.shape # get width and height
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            # get rows i -> i + patch_size (not inclusive)
            # get colums j -> j + patch_size (not inclusive)
            patch = img[i:i+patch_size, j:j+patch_size]
            # make sure that the patch is square
            if patch.shape == (patch_size, patch_size):
                patches.append(patch)
    return patches

def compute_patch_results(patch, n, cell_size, blob_size, alpha):
    # true coefs of theta
    coeffs_true = generate_coeff_vector( patch, n, cell_size, blob_size)

    # V1 - SVD
    measurement_matrix_V1, V1_y = generate_V1_observation(patch, n, cell_size, blob_size, None)
    theta_V1 = generate_design_matrix(measurement_matrix_V1)
    U_V1, S_V1, Vh_V1 = np.linalg.svd(theta_V1)

    # V1 - estimated coeffs
    reconst_v1 = reconstruct(measurement_matrix_V1, V1_y, alpha)
    coeffs_est_V1 = generate_coeff_vector(reconst_v1, n, cell_size, blob_size)

    # V1 - PCs
    a_est_V1 = Vh_V1 @ coeffs_est_V1.flatten()
    a_true_V1 = Vh_V1 @ coeffs_true.flatten()

    err_V1 = (a_true_V1 - a_est_V1) ** 2

    # Pixel - SVD
    measurement_matrix_pix, pixel_y = generate_pixel_observation(patch, n)
    theta_pix = generate_design_matrix(measurement_matrix_pix)
    U_pix, S_pix, Vh_pix = np.linalg.svd(theta_pix)

    # Pixel - estimated coeffs
    reconst_pix = reconstruct(measurement_matrix_pix, pixel_y, alpha)
    coeffs_est_pix = generate_coeff_vector(reconst_pix, n, cell_size, blob_size)

    # Pixel - PCs
    a_est_pix = Vh_pix @ coeffs_est_pix.flatten()
    a_true_pix = Vh_pix @ coeffs_true.flatten()

    err_pix = (a_true_pix - a_est_pix) ** 2

    # Gauss - SVD
    measurement_matrix_gauss, gaussian_y = generate_gaussian_observation(patch, n)
    theta_gauss = generate_design_matrix(measurement_matrix_gauss)
    U_gauss, S_gauss, Vh_gauss = np.linalg.svd(theta_gauss)

    # Gauss - estimated coeffs
    reconst_gauss = reconstruct(measurement_matrix_gauss, gaussian_y, alpha)
    coeffs_est_gauss = generate_coeff_vector(reconst_gauss, n, cell_size, blob_size)

    # Gauss - PCs
    a_est_gauss = Vh_gauss @ coeffs_est_gauss.flatten()
    a_true_gauss = Vh_gauss @ coeffs_true.flatten()

    err_gauss = (a_true_gauss - a_est_gauss) ** 2

    return {
        "coeffs_true": coeffs_true,

        "V1": {
            "U" : U_V1,
            "S" : S_V1,
            "Vh": Vh_V1,
            "reconstruction": reconst_v1,
            "est_coeffs" : coeffs_est_V1,
            "a_est": a_est_V1,
            "a_true": a_true_V1,
            "error" : err_V1,
        },
        "Pixel": {
            "U" : U_pix,
            "S" : S_pix,
            "Vh": Vh_pix,
            "reconstruction": reconst_pix,
            "est_coeffs" : coeffs_est_pix,
            "a_est": a_est_pix,
            "a_true": a_true_pix,
            "error" : err_pix,
        },
        "Gaussian": {
            "U" : U_gauss,
            "S" : S_gauss,
            "Vh": Vh_gauss,
            "reconstruction": reconst_gauss,
            "est_coeffs" : coeffs_est_gauss,
            "a_est": a_est_gauss,
            "a_true": a_true_gauss,
            "error" : err_gauss,
        }
    }

def get_results(img):
    patches = extract_patches(img, PATCH_SIZE)
    print(len(patches))
    all_results = []
    for patch in patches:
        res = compute_patch_results(patch, N_OBS, CELL_SIZE, BLOB_SIZE, ALPHA)
        all_results.append(res)

    return all_results

boat = process_image("boat.png", False)
print(boat.shape)
results = get_results(boat)


