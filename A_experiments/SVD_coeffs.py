import os
import sys
import numpy as npS
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import sys
import seaborn as sns
import pandas as pd


sys.path.append('..')
from src.compress_sensing import *
from src.utility import *
from A_experiments.theta_exp_improved import *


'''
Compare the Fourier coefficients to singular values of theta.
'''
# Find the true coefficients of theta
coeffs_true = generate_coeff_vector(small_img_arr_gray, num_cell_300, cell_size, blob_size)
U_c_true, S_c_true, Vh_c_true = np.linalg.svd(coeffs_true)

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
#U_c_V1, S_c_V1, Vh_c_V1 = np.linalg.svd(coeffs_est_V1)
a_est_V1 = Vh_V1 @ coeffs_est_V1.flatten()
a_true_V1 = Vh_V1 @ coeffs_true.flatten()

#U_c_pix, S_c_pix, Vh_c_pix = np.linalg.svd(coeffs_est_pix)
a_est_pix = Vh_pix @ coeffs_est_pix.flatten()
a_true_pix = Vh_pix @ coeffs_true.flatten()

#U_c_gauss, S_c_gauss, Vh_c_gauss = np.linalg.svd(coeffs_est_gauss)
a_est_gauss = Vh_gauss @ coeffs_est_gauss.flatten()
a_true_gauss = Vh_gauss @ coeffs_true.flatten()


# compare squared error for principal components and raw pixel values

squared_error_V1 = (a_true_V1 - a_est_V1) ** 2
mse_V1 = np.mean(squared_error_V1)


im = process_image(small_img,False)
true_pixels = np.array(im)
true_pixels = true_pixels.reshape(1,900)

v1_pixels = reconst_gray_300_v1.reshape(1,900)

pixel_squared_error = np.empty(900)
for i in range(900):
    pixel_squared_error[i] = (true_pixels[0,i] - v1_pixels[0,i]) ** 2

mean_pixel_error = np.mean(pixel_squared_error)

print("PC mse_V1:", mse_V1)
print("Pixel mse:", mean_pixel_error)


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
    
    #plt.savefig(filename)
    #plt.close()


#make_scatter(a_est_V1,   a_true_V1, "V1", "V1 vs True (300 samples)", "V1_vs_True_300_YlOrRd.png", alpha=0.5)
#make_scatter(a_est_pix,  a_true_pix, "Pixel", "Pixel vs True (300 samples)", "Pixel_vs_True_300_YlOrRd.png")
#make_scatter(a_est_gauss, a_true_gauss, "Gaussian", "Gaussian vs True (300 samples)", "Gaussian_vs_True_300_YlOrRd.png")

def scatter_PCs(components, xlabel, title):
    plt.figure(figsize=(10,8), dpi=200)
    sc = plt.scatter([i for i in range(900)], components)

    plt.xlabel("Rank")
    plt.ylabel(f"{xlabel} Principal Component")
    #plt.xscale('log')
    plt.yscale('log')
    plt.title(title)

#scatter_PCs(a_est_V1, "V1", "V1 (300 samples)")
#scatter_PCs(a_true_V1, "V1 True", "V1 True (300 samples)")
#scatter_PCs(a_est_pix, "Pixel", "Pixel (300 samples)")
#scatter_PCs(a_est_gauss, "Gaussian", "Gaussian (300 samples)")


def plot_errors(errors, xlabel,title):
    plt.figure(figsize=(10,8), dpi=200)
    sc = plt.plot([i for i in range(900)], errors)

    plt.xlabel(f"{xlabel} Squared Error")
    plt.ylabel("Index")
    plt.xscale('log')
    plt.yscale('log')
    plt.title(title)

plt.figure(figsize=(10,8), dpi=200)
plt.plot([i for i in range(900)], (a_true_V1 - a_est_V1) ** 2, label="V1")
plt.plot([i for i in range(900)], (a_true_pix - a_est_pix) ** 2, label="Pixel")
plt.plot([i for i in range(900)], (a_true_gauss - a_est_gauss) ** 2, label="Gaussian")

plt.xlabel("Squared Error")
plt.legend()
plt.ylabel("Index")
plt.xscale('log')
plt.yscale('log')
plt.title("Error Per Component (300 Samples)")
#plt.savefig("error_300.png")


# compute squared error
err_v1 = (a_true_V1 - a_est_V1) ** 2
err_pix = (a_true_pix - a_est_pix) ** 2
err_gauss = (a_true_gauss - a_est_gauss) ** 2

# wide data frame
wide_df = pd.DataFrame({
    'Index': range(900),
    'V1': err_v1,
    'Pixel': err_pix,
    'Gaussian': err_gauss
})

# line plot is optimized for "long" format for plotting multiple lines: one column for the category (Method) and one for the values (Squared Error)
# .melt reshapes table from wide -> long. Each row is a single observation w/ its method label
df_long = wide_df.melt(id_vars='Index', var_name='Method', value_name='Squared Error')

# rolling mean (window of 150 components) to smooth the curve
# groupby -> each observation type is seperate
# min_periods=1 prevents NaN values at the start of the series.
df_long['Smoothed Error'] = df_long.groupby('Method')['Squared Error'].transform(
    lambda x: x.rolling(window=150, min_periods=1).mean()
)

plt.figure(figsize=(10, 8), dpi=200)
sns.lineplot(data=df_long, x='Index', y='Smoothed Error', hue='Method')

plt.xscale('log')
plt.yscale('log')
plt.title("Error Per Component (300 Obs)")
plt.xlabel("Index")
plt.ylabel("Squared Error")

#plt.savefig("smoothed_error_plot_300.png", dpi=300, bbox_inches='tight')


# '''
# Reconstructions as pics

# '''

reconstructed_images = {
    "V1": reconst_gray_300_v1,
    "Pixel": reconst_gray_300_pix,
    "Gaussian": reconst_gray_300_gauss
}

plt.figure(figsize=(12,4))

for i, (label, img_flat) in enumerate(reconstructed_images.items()):
    plt.subplot(1, 3, i+1)
    plt.imshow(img_flat.reshape(30,30), cmap='gray')
    plt.title(f'Reconstructed ({label})')
    plt.axis('off')

plt.suptitle("Reconstructed Images (300 obs)", fontsize=15)
plt.tight_layout()
#plt.savefig("reconstructed_images_300.png")
plt.close()

'''
PCs as pics

'''

pcs = {
    "V1": a_est_V1,
    "Pixel": a_est_pix,
    "Gaussian": a_est_gauss
}
plt.figure(figsize=(12,4))

for i, (label, img_flat) in enumerate(pcs.items()):
    plt.subplot(1, 3, i+1)
    plt.imshow(img_flat.reshape(30,30), cmap='gray')
    plt.title(f'Reconstructed ({label})')
    plt.axis('off')

plt.suptitle("Reconstructed Images from PC (300 Obs)", fontsize=15)
plt.tight_layout()
#plt.savefig("pc_images_300.png")
plt.close()

'''
Sparcity of coeffs vectors - histogram of entries in coeffs vectors

'''

coeff_vectors = {
    "True": coeffs_true.flatten(),
    "V1 Estimated": coeffs_est_V1.flatten(),
    "Pixel Estimated": coeffs_est_pix.flatten(),
    "Gaussian Estimated": coeffs_est_gauss.flatten()
}

bins = np.logspace(-1, 1, 50) #bins for histogram

plt.figure(figsize=(16, 4))

for i, (label, coeffs) in enumerate(coeff_vectors.items()):
    plt.subplot(1, 4, i+1)
    plt.hist(np.abs(coeffs), bins=bins, alpha=0.7, color='C'+str(i))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Absolute Coefficient Value")
    plt.ylabel("Number of Coefficients")
    plt.title(label)
    plt.grid(True, which="both", ls="--", lw=0.5)

plt.suptitle("Sparsity of Coefficient Vectors (300)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig("coeffs_sparsity_comparison_300.png")
plt.close()