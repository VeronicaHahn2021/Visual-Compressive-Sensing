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


'''
Plot the spectrum of theta. Measure of sparsity?
'''

# compute singular values for a given number of observations
def compute_singular_values(num_cell):
    # V1
    measurement_matrix_V1, V1_y_300 = generate_V1_observation(small_img_arr_gray, num_cell, cell_size, blob_size, None)
    theta_V1 = generate_design_matrix(measurement_matrix_V1)
    U, S_V1, V = np.linalg.svd(theta_V1)
    
    # pix
    measurement_matrix_pix, V1_y_300 = generate_pixel_observation(small_img_arr_gray, num_cell)
    theta_pix = generate_design_matrix(measurement_matrix_pix)
    U, S_pix, V = np.linalg.svd(theta_pix)
    
    # gauss
    measurement_matrix_gauss, V1_y_300 = generate_gaussian_observation(small_img_arr_gray, num_cell)
    theta_gauss = generate_design_matrix(measurement_matrix_gauss)
    U, S_gauss, V = np.linalg.svd(theta_gauss)
    
    return S_V1, S_pix, S_gauss

# compute SVD
S_V1_100, S_pix_100, S_gauss_100 = compute_singular_values(num_cell_100)
S_V1_300, S_pix_300, S_gauss_300 = compute_singular_values(num_cell_300)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # width x height

# 100 obs
axes[0].plot(np.arange(1, num_cell_100+1), S_V1_100, "o", label="V1")
axes[0].plot(np.arange(1, num_cell_100+1), S_pix_100, "x", label="Pix")
axes[0].plot(np.arange(1, num_cell_100+1), S_gauss_100, "+", label="Gauss")
axes[0].set_title("SVD (100 Observations)")
axes[0].set_xlabel("index")
axes[0].set_ylabel("Singular Value")
axes[0].legend()

# 300 obs
axes[1].plot(np.arange(1, num_cell_300+1), S_V1_300, "o", label="V1")
axes[1].plot(np.arange(1, num_cell_300+1), S_pix_300, "x", label="Pix")
axes[1].plot(np.arange(1, num_cell_300+1), S_gauss_300, "+", label="Gauss")
axes[1].set_title("SVD (300 Observations)")
axes[1].set_xlabel("index")
axes[1].set_ylabel("Singular Value")
axes[1].legend()

plt.tight_layout()
plt.savefig("singular_values_100_vs_300.svg", format="svg")
plt.show()

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