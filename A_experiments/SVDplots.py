import os
import sys
import numpy as npS
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import sys

sys.path.append('..')
from src.compress_sensing import *
from src.utility import *
from A_experiments.theta_exp_improved import *
from A_experiments.paper_aligned_plots import *

PATCH_SIZE = 32
CELL_SIZE = 50
BLOB_SIZE = 6
ALPHA = 1

PATCH_IDXS = [ 58, 169, 206, 233]
NUM_OBS_LIST = [256]

img = process_image("barbara.bmp", color=False)
patches = extract_patches(img, PATCH_SIZE)

def compute_patch_singular_values(patch, num_cell):

    # V1
    W_v1, _ = generate_V1_observation(
        patch, num_cell, CELL_SIZE, BLOB_SIZE, None
    )
    theta_v1 = generate_design_matrix(W_v1)
    _, S_v1, _ = np.linalg.svd(theta_v1, full_matrices=False)

    # Pixel
    W_pix, _ = generate_pixel_observation(patch, num_cell)
    theta_pix = generate_design_matrix(W_pix)
    _, S_pix, _ = np.linalg.svd(theta_pix, full_matrices=False)

    # Gaussian
    W_gauss, _ = generate_gaussian_observation(patch, num_cell)
    theta_gauss = generate_design_matrix(W_gauss)
    _, S_gauss, _ = np.linalg.svd(theta_gauss, full_matrices=False)

    return S_v1, S_pix, S_gauss


'''
Plot the spectrum of theta. Measure of sparsity?
'''

# compute singular values for a given number of observations
def compute_singular_values(num_cell):
    # V1
    measurement_matrix_V1, V1_y = generate_V1_observation(small_img_arr_gray, num_cell, cell_size, blob_size, None)
    theta_V1 = generate_design_matrix(measurement_matrix_V1)
    U, S_V1, V = np.linalg.svd(theta_V1)
    
    # pix
    measurement_matrix_pix, V1_y = generate_pixel_observation(small_img_arr_gray, num_cell)
    theta_pix = generate_design_matrix(measurement_matrix_pix)
    U, S_pix, V = np.linalg.svd(theta_pix)
    
    # gauss
    measurement_matrix_gauss, V1_y = generate_gaussian_observation(small_img_arr_gray, num_cell)
    theta_gauss = generate_design_matrix(measurement_matrix_gauss)
    U, S_gauss, V = np.linalg.svd(theta_gauss)
    
    return S_V1, S_pix, S_gauss

# compute SVD
S_V1_100, S_pix_100, S_gauss_100 = compute_singular_values(num_cell_100)
S_V1_300, S_pix_300, S_gauss_300 = compute_singular_values(num_cell_300)


def plot_SVD(num_plots, patches, savefile):
    fig, axes = plt.subplots(1, len(PATCH_IDXS), figsize=(18, 6))
    i = 0
    for patch_idx in PATCH_IDXS:
        S_V1_256_, S_pix_256_, S_gauss_256_ = compute_patch_singular_values(patches[patch_idx], 256)
        axes[i].plot(np.arange(1, num_plots+1), S_V1_256_, "o", label="V1")
        axes[i].plot(np.arange(1, num_plots+1), S_pix_256_, "x", label="Pix")
        axes[i].plot(np.arange(1, num_plots+1), S_gauss_256_, "+", label="Gauss")
        axes[i].set_title(f"Patch {patch_idx}")
        axes[i].set_xlabel("Index")
        axes[i].set_ylabel("Singular Value")
        axes[i].legend()
        i += 1
    plt.suptitle(f"SVD", fontsize=16)
    plt.tight_layout()
    plt.savefig(savefile)
    plt.show()

'''
put in 2 rows
'''
def plot_SVD(num_plots, patches, savefile):

 # TODO: print out the pixel values for this plot to see if the pixel values in the largest step is actually all the same
 # TODO: for kameron, see if we can plot this as a distribution, same column 
    n_patches = len(PATCH_IDXS)
    ncols = int(np.ceil(n_patches / 2))
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 8), sharey=True)
    axes = axes.flatten()  # makes indexing easy

    for i, patch_idx in enumerate(PATCH_IDXS):

        S_V1_256_, S_pix_256_, S_gauss_256_ = compute_patch_singular_values(patches[patch_idx], 256)

        axes[i].plot(np.arange(1, num_plots+1), S_V1_256_, "o", label="V1")
        axes[i].plot(np.arange(1, num_plots+1), S_pix_256_, "x", label="Pix")
        axes[i].plot(np.arange(1, num_plots+1), S_gauss_256_, "+", label="Gauss")

        axes[i].set_title(f"Patch {patch_idx}")
        if (i == 2 or i == 3):
            axes[i].set_xlabel("Index")
        if (i == 0 or i == 2):
            axes[i].set_ylabel("Singular Value")
        if (patch_idx == 169 or patch_idx == 235):
            axes[i].tick_params(axis='y', which='both', left=False, labelleft=False)
        axes[i].legend()
        print(i)

    for j in range(n_patches, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"SVD", fontsize=16)
    plt.tight_layout()
    plt.savefig(savefile)
    plt.show()


#plot_SVD(256, patches, "SVD_256_patches.svg")

def plot_SVD_single_patch(num_plots, patches, savefile):

    patch_idx = 58  # explicitly choose patch 58

    S_V1_256_, S_pix_256_, S_gauss_256_ = \
        compute_patch_singular_values(patches[patch_idx], 256)

    plt.figure(figsize=(8, 6))

    plt.plot(np.arange(1, num_plots+1), S_V1_256_, "o", label="V1")
    plt.plot(np.arange(1, num_plots+1), S_pix_256_, "x", label="Pix")
    plt.plot(np.arange(1, num_plots+1), S_gauss_256_, "+", label="Gauss")

    plt.title(f"SVD - Patch {patch_idx}")
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.legend()

    plt.tight_layout()
    plt.savefig(savefile)
    plt.show()

plot_SVD_single_patch(256, patches, "SVD_patch_58.svg")

'''
OG code 
'''
# # Find the singular values of theta
# measurement_matrix_V1, V1_y_300 = generate_V1_observation(small_img_arr_gray, num_cell_300, cell_size, blob_size, None)
# theta_V1 = generate_design_matrix(measurement_matrix_V1)

# U, S_V1, V = np.linalg.svd(theta_V1)

# measurement_matrix_pix, V1_y_300 = generate_pixel_observation(small_img_arr_gray, num_cell_300)
# theta_pix = generate_design_matrix(measurement_matrix_pix)

# U, S_pix, V = np.linalg.svd(theta_pix)

# measurement_matrix_gauss, V1_y_300 = generate_gaussian_observation(small_img_arr_gray, num_cell_300)
# theta_gauss = generate_design_matrix(measurement_matrix_gauss)

# U, S_gauss, V = np.linalg.svd(theta_gauss)

# rank_V1 = np.count_nonzero(S_V1) 
# rank_pix = np.count_nonzero(S_pix)
# rank_gauss = np.count_nonzero(S_gauss)

# plt.figure()
# plt.plot(np.arange(1, 300+1), S_V1, "o", label="V1")
# plt.plot(np.arange(1, 300+1), S_pix, "x", label="Pix")
# plt.plot(np.arange(1, 300+1), S_gauss, "+", label="Gauss")
# plt.legend()
# plt.title("Singular Values of Theta")
# plt.xlabel("jth Column")
# plt.ylabel("Value")
# plt.show()