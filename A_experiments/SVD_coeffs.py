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
a_true = np.reshape(V_c_true.T * coeffs_true,-1)


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


# find principal components from coefficient vectors for each obs type
U_c_V1, S_c_V1, V_c_V1 = np.linalg.svd(coeffs_est_V1)
a_est_V1 = np.reshape(V_c_V1 * coeffs_est_V1,-1)

U_c_pix, S_c_pix, V_c_pix = np.linalg.svd(coeffs_est_pix)
a_est_pix = np.reshape(V_c_pix * coeffs_est_pix,-1)

U_c_gauss, S_c_gauss, V_c_gauss = np.linalg.svd(coeffs_est_gauss)
a_est_gauss = np.reshape(V_c_gauss * coeffs_est_gauss,-1)

# Plot principal components

'''
plt.figure()
sc = plt.scatter(np.abs(a_est_V1), np.abs(a_true), c=[i for i in range(900)])
plt.colorbar(sc)

plt.ylabel("true principal component")
plt.xlabel("V1 principal component")
plt.yscale('log')
plt.xscale('log')
plt.title("Comparing V1 to True")

plt.show()
'''

'''
plt.figure()
plt.plot([i for i in range(900)], np.abs(a_true), '.', label = "true")


plt.ylabel("Principal Component")
plt.xlabel("i")
plt.yscale('log')
plt.title("Principal Component for True Coefficients (300 observations)")
plt.ylim(bottom=10 ** (-6))
plt.show()


plt.figure()
plt.plot([i for i in range(900)], np.abs(a_est_V1), 'x', label = "V1")
plt.ylabel("Principal Component")
plt.xlabel("i")
plt.yscale('log')
plt.title("Principal Component for V1 Coefficients (300 observations)")
plt.ylim(bottom=10 ** (-6))
plt.show()

plt.figure()
plt.plot([i for i in range(900)], np.abs(a_est_pix), '+',  label = "pix")
plt.ylabel("Principal Component")
plt.xlabel("i")
plt.yscale('log')
plt.title("Principal Component for Pix Coefficients (300 observations)")
plt.ylim(bottom=10 ** (-6))
plt.show()

plt.figure()
plt.plot([i for i in range(900)], np.abs(a_est_gauss), '*', label = "gauss")
plt.ylabel("Principal Component")
plt.xlabel("i")
plt.yscale('log')
plt.title("Principal Component for Gaussian Coefficients (300 observations)")
plt.ylim(bottom=10 ** (-6))
plt.show()
'''


# compare squared error for principal components and raw pixel values

def compute_squared_error(estimated, true):
    '''Do this stuff'''

    squared_error = np.empty(900)

    for i in range(900):
        squared_error[i] = (true[i] - estimated[i]) ** 2

    return squared_error

# plot squared error

plt.figure()
plt.plot([i for i in range(900)], compute_squared_error(a_est_V1, a_true), label="V1")
plt.plot([i for i in range(900)], compute_squared_error(a_est_gauss, a_true), label="Gaussian")
plt.plot([i for i in range(900)], compute_squared_error(a_est_pix, a_true), label="pix")
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.title("Error per component")
plt.xlabel("index i")
plt.ylabel("(a_true - a_est)^2")
plt.show()


mse = np.sum(compute_squared_error(a_est_V1,a_true)) / 900

im = process_image(small_img,False)
true_pixels = np.array(im)
true_pixels = true_pixels.reshape(1,900)

v1_pixels = reconst_gray_300_v1.reshape(1,900)

pixel_squared_error = np.empty(900)
for i in range(900):
    pixel_squared_error[i] = (true_pixels[0,i] - v1_pixels[0,i]) ** 2

mean_pixel_error = np.sum(pixel_squared_error) / 900


# Energy of the image
energy = np.sum(true_pixels ** 2) / 900