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
U_c_true, S_c_true, V_c_true = np.linalg.svd(coeffs_true)
a_true = V_c_true.T * coeffs_true


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


reconst_gray_300_pix = reconstruct(measurement_matrix_pix, pixel_y_300, alpha)
coeffs_est_pix = generate_coeff_vector(reconst_gray_300_pix, num_cell_300, cell_size, blob_size)


reconst_gray_300_gauss = reconstruct(measurement_matrix_gauss, gaussian_y_300, alpha)
coeffs_est_gauss = generate_coeff_vector(reconst_gray_300_gauss, num_cell_300, cell_size, blob_size)


# find a_i from coefficient vectors from each obs type
U_c_V1, S_c_V1, V_c_V1 = np.linalg.svd(coeffs_est_V1)
a_est_V1 = V_c_V1.T * coeffs_est_V1

U_c_pix, S_c_pix, V_c_pix = np.linalg.svd(coeffs_est_pix)
a_est_pix = V_c_pix.T * coeffs_est_pix

U_c_gauss, S_c_gauss, V_c_gauss = np.linalg.svd(coeffs_est_gauss)
a_est_gauss = V_c_gauss.T * coeffs_est_gauss

# Plot a_i
plt.figure()

k=29
plt.plot([i for i in range(30)], a_true[k], '.', label = "true")
plt.plot([i for i in range(30)], a_est_V1[k], 'x', label = "V1")
plt.plot([i for i in range(30)], a_est_pix[k], '+',  label = "pix")
plt.plot([i for i in range(30)], a_est_gauss[k], '*', label = "gauss")


plt.ylabel("a_i")
plt.xlabel("i")
plt.title("a_i for row " + str(k))
plt.legend()
plt.show()

# Plot for V1 and true only
plt.figure()

plt.plot([i for i in range(30)], a_true[k], '.', label = "true")
plt.plot([i for i in range(30)], a_est_V1[k], 'x', label = "V1")

plt.ylabel("a_i")
plt.xlabel("i")
plt.title("a_i for row " + str(k))
plt.legend()
plt.show()

'''
plt.figure()
plt.imshow(a_true)
plt.colorbar()
plt.title("a_i heatmap")'''