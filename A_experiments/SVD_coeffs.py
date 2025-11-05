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
Compare the Fourier coefficients to singular values of theta.
'''
# Find the true coefficients of theta
coeffs_true = generate_coeff_vector(small_img_arr_gray, num_cell_300, cell_size, blob_size)


# Find the singular values of theta
measurement_matrix_V1, V1_y_300 = generate_V1_observation(small_img_arr_gray, num_cell_300, cell_size, blob_size, None)
theta_V1 = generate_design_matrix(measurement_matrix_V1)

U_V1, S_V1, V_V1 = np.linalg.svd(theta_V1)


measurement_matrix_pix, pixel_y_300 = generate_pixel_observation(small_img_arr_gray, num_cell_300)
theta_pix = generate_design_matrix(measurement_matrix_pix)

U_pix, S_pix, V_pix = np.linalg.svd(theta_pix)

measurement_matrix_gauss, gaussian_y_300 = generate_gaussian_observation(small_img_arr_gray, num_cell_300)
theta_gauss = generate_design_matrix(measurement_matrix_gauss)

U_gauss, S_gauss, V_gauss = np.linalg.svd(theta_gauss)

# Find the estimated coefficients for each observation type
reconst_gray_300_v1 = reconstruct(measurement_matrix_V1, V1_y_300, alpha)
coeffs_est_V1 = generate_coeff_vector(reconst_gray_300_v1, num_cell_300, cell_size, blob_size)
a_est_V1 = V_V1.T * coeffs_est_V1

reconst_gray_300_pix = reconstruct(measurement_matrix_pix, pixel_y_300, alpha)
coeffs_est_pix = generate_coeff_vector(reconst_gray_300_pix, num_cell_300, cell_size, blob_size)
a_est_pix = V_pix.T * coeffs_est_pix

reconst_gray_300_gauss = reconstruct(measurement_matrix_gauss, gaussian_y_300, alpha)
coeffs_est_gauss = generate_coeff_vector(reconst_gray_300_gauss, num_cell_300, cell_size, blob_size)
a_est_gauss = V_gauss.T * coeffs_est_gauss

