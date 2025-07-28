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
Plot the coefficient vectors for an image and its V1 reconstruction.
'''

coeffs_true = generate_coeff_vector(small_img_arr_gray, num_cell_300, cell_size, blob_size)
bins=100

V1_W_300, V1_y_300 = generate_V1_observation(small_img_arr_gray, num_cell_300, cell_size, blob_size, None)
'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
# ## Reconstruction with 300 number of cells grayscaled
ax1.imshow(small_img_arr_gray,'gray')
ax1.set_title("Original")
ax1.axis("off")

reconst_gray_300 = reconstruct(V1_W_300, V1_y_300, alpha)
ax2.imshow(reconst_gray_300, 'gray')
ax2.set_title("Reconstruct{num_cell} number of cells".format(num_cell = num_cell_300))
ax2.axis("off")
plt.subplots_adjust(top=1.35)
plt.show()
'''

reconst_gray_300 = reconstruct(V1_W_300, V1_y_300, alpha)
coeffs_est = generate_coeff_vector(reconst_gray_300, num_cell_300, cell_size, blob_size)



plt.figure()
plt.hist(np.abs(coeffs_true).flatten(), bins, cumulative = False, density = True, label = "True")
plt.hist(np.abs(coeffs_est).flatten(), bins, cumulative = False, density = True, label = "Estimated")
plt.xlabel("Coefficient Vectors")
plt.ylabel("Frequency")
plt.legend()
plt.yscale('log')
plt.show()
plt.savefig("Coeffs Hist")

plt.figure()
plt.title("True")
plt.imshow(coeffs_true)
plt.colorbar()

plt.figure()
plt.title("Estimated")
plt.imshow(coeffs_est)
plt.colorbar()