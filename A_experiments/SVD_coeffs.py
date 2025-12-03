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
# U_c_true, S_c_true, Vh_c_true = np.linalg.svd(coeffs_true)


# Find the singular values of theta
measurement_matrix_V1, V1_y_300 = generate_V1_observation(small_img_arr_gray, num_cell_300, cell_size, blob_size, None)
theta_V1 = generate_design_matrix(measurement_matrix_V1)

U_V1, S_V1, Vh_V1 = np.linalg.svd(theta_V1)

measurement_matrix_pix, pixel_y_300 = generate_pixel_observation(small_img_arr_gray, num_cell_300)
theta_pix = generate_design_matrix(measurement_matrix_pix)

U_pix, S_pix, Vh_pix = np.linalg.svd(theta_pix)

measurement_matrix_gauss, gaussian_y_300 = generate_gaussian_observation(small_img_arr_gray, num_cell_300)
theta_gauss = generate_design_matrix(measurement_matrix_gauss)

U_gauss, S_gauss, Vh_gauss = np.linalg.svd(theta_gauss)

# Find the estimated coefficients for each observation type
reconst_gray_300_v1 = reconstruct(measurement_matrix_V1, V1_y_300, alpha)
coeffs_est_V1 = generate_coeff_vector(reconst_gray_300_v1, num_cell_300, cell_size, blob_size)

reconst_gray_300_pix = reconstruct(measurement_matrix_pix, pixel_y_300, alpha)
coeffs_est_pix = generate_coeff_vector(reconst_gray_300_pix, num_cell_300, cell_size, blob_size)


reconst_gray_300_gauss = reconstruct(measurement_matrix_gauss, gaussian_y_300, alpha)
coeffs_est_gauss = generate_coeff_vector(reconst_gray_300_gauss, num_cell_300, cell_size, blob_size)


# find principal components from coefficient vectors for each obs type
# get sensing matrix out out these and run again (must be 300 x 900)
# U_c_V1, S_c_V1, Vh_c_V1 = np.linalg.svd(coeffs_est_V1) # run this w/ measurement_matrix_V1
a_est_V1 = Vh_V1 @ coeffs_est_V1.flatten()


# U_c_pix, S_c_pix, Vh_c_pix = np.linalg.svd(coeffs_est_pix)
a_est_pix = Vh_pix @ coeffs_est_pix.flatten()

# U_c_gauss, S_c_gauss, Vh_c_gauss = np.linalg.svd(coeffs_est_gauss)
a_est_gauss = Vh_gauss @ coeffs_est_gauss.flatten()

# Plot principal components

def make_scatter(est, true, xlabel, title, filename, figsize=(10, 8), dpi=200, marker_size=30, cmap='YlOrRd', alpha=0.5):
    plt.figure(figsize=figsize, dpi=dpi)
    sc = plt.scatter(np.abs(est), np.abs(true), 
                     c=np.arange(len(est)), s=marker_size,
                     cmap=cmap, alpha=alpha)
    
    plt.colorbar(sc).set_label('PC rank', rotation=270, labelpad=15) # add label to colorbar

    plt.xlabel(f"{xlabel} Principal Component")
    plt.ylabel("True Principal Component")
    plt.xscale('log')
    plt.yscale('log')
    plt.title(title)

    # add y=x line
    # visibal range on axes
    xmin, xmax = plt.xlim() # min, max of x
    ymin, ymax = plt.ylim() # min, max of y
    low = max(xmin, ymin) # start at largest of 2 mins, so it doesn't go below
    high = min(xmax, ymax) # end at smallest of 2 maxima -> doesn't go beyond
    plt.plot([low, high], [low, high])
    
    plt.savefig(filename)
    plt.close()

a_true_V1 = Vh_V1 @ coeffs_true.flatten()

#make_scatter(a_est_V1,   a_true, "V1", "V1 vs True (100 samples)", "V1_vs_True_100_YlOrRd.png", alpha=0.5)
#make_scatter(a_est_pix,  a_true, "Pixel", "Pixel vs True (100 samples)", "Pixel_vs_True_100_YlOrRd.png")
#make_scatter(a_est_gauss, a_true, "Gaussian", "Gaussian vs True (100 samples)", "Gaussian_vs_True_100_YlOrRd.png")


# # mean squared error
# squared_error = np.empty(900)

# for i in range(900):
#     squared_error[i] = (a_true[i] - a_est_V1[i]) ** 2

squared_error = (a_true_V1 - a_est_V1) ** 2

mse = np.mean(squared_error)

im = process_image(small_img,False)
true_pixels = np.array(im)
true_pixels = true_pixels.reshape(1,900)

v1_pixels = reconst_gray_300_v1.reshape(1,900)

pixel_squared_error = np.empty(900)
for i in range(900):
    pixel_squared_error[i] = (true_pixels[0,i] - v1_pixels[0,i]) ** 2

mean_pixel_error = np.sum(pixel_squared_error) / 900

print("PC MSE:", mse)
print("Pixel MSE:", mean_pixel_error)


