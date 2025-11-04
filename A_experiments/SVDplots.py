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

#Graph the singular values of theta
measurement_matrix_V1, V1_y_300 = generate_V1_observation(small_img_arr_gray, num_cell_300, cell_size, blob_size, None)
theta_V1 = generate_design_matrix(measurement_matrix_V1)

U, S_V1, V = np.linalg.svd(theta_V1)

measurement_matrix_pix, V1_y_300 = generate_pixel_observation(small_img_arr_gray, num_cell_300)
theta_pix = generate_design_matrix(measurement_matrix_pix)

U, S_pix, V = np.linalg.svd(theta_pix)

measurement_matrix_gauss, V1_y_300 = generate_gaussian_observation(small_img_arr_gray, num_cell_300)
theta_gauss = generate_design_matrix(measurement_matrix_gauss)

U, S_gauss, V = np.linalg.svd(theta_gauss)

rank_V1 = np.count_nonzero(S_V1) 
rank_pix = np.count_nonzero(S_pix)
rank_gauss = np.count_nonzero(S_gauss)

'''
plt.figure()
plt.hist(S_V1, bins=100, label="V1")
plt.hist(S_pix, bins=100, label="pix")
plt.hist(S_gauss, bins=100, label="gaussian")
plt.legend()
plt.title("Singular Value Distribution")
plt.xlabel("Singular Value")
plt.ylabel("Distribution")
plt.yscale("log")
plt.show()
'''

plt.figure()
plt.plot(np.arange(1, 300+1), S_V1, "o", label="V1")
plt.plot(np.arange(1, 300+1), S_pix, "x", label="Pix")
plt.plot(np.arange(1, 300+1), S_gauss, "+", label="Gauss")
plt.legend()
plt.title("Singular Values of Theta")
plt.xlabel("jth Column")
plt.ylabel("Value")
plt.show()
