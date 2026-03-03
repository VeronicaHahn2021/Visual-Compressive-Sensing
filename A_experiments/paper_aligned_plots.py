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
CELL_SIZE = 50
BLOB_SIZE = 6
'''
58 = eye
169 = pattern
206 = stain 1
235 = stain 2
'''
PATCH_IDXS = [ 58]

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

def show_patches_grid(patches, cols=16):
    n_patches = len(patches)
    rows = (n_patches + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten()

    for ax, patch in zip(axes, patches):
        if patch.ndim == 2:
            # TODO: set color bar to be the same on all patches to get rid of weird gray scale
            ax.imshow(patch, cmap='gray')
        else:
            ax.imshow(patch)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    fig.savefig("patches_16.png")

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
    all_results = []
    for patch in patches:
        res = compute_patch_results(patch, N_OBS, CELL_SIZE, BLOB_SIZE, ALPHA)
        all_results.append(res)
    return all_results

def run_selected_patches(patches, patch_idxs):
    results = {}

    for idx in patch_idxs:
        print(f"Running patch {idx}")
        results[idx] = compute_patch_results(
            patches[idx],
            N_OBS,
            CELL_SIZE,
            BLOB_SIZE,
            ALPHA
        )

    return results

barbara = process_image("barbara.bmp", color=False)
patches = extract_patches(barbara, PATCH_SIZE)
#show_patches_grid(patches)
results = run_selected_patches(patches, PATCH_IDXS)

'''
PC's as scatter plots - together
'''
def pc_scatter_plots(results, num_obs, filename, patch_idx, cmap='cool',):

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, method in zip(axes, ["V1", "Pixel", "Gaussian"]):
        # get estimate and true PCs
        est = results[num_obs][method]["a_est"]
        true = results[num_obs][method]["a_true"]

        # make scatter plot
        sc = ax.scatter(np.abs(est), np.abs(true), c=np.arange(len(est)), s = 30, cmap=cmap, alpha=0.5)

        # y = x line
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        low = max(xmin, ymin) # start at largest of the 2 mins, so it doesn't go below
        high = min(xmax, ymax) # end at smallest of 2 maxima -> doesn't go beyond
        ax.plot([low, high], [low, high], '--', color='gray')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"{method} vs True")
        ax.set_xlabel("Estimated Principal Component")
        ax.set_ylabel("True Principal Component")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('PC rank', rotation=270, labelpad=15)

    plt.suptitle(f"True vs Estimated Principal Components - Patch {patch_idx}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def pc_per_method(results, num_obs, patch_idx):
    
    methods = ["V1", "Pixel", "Gaussian"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, method in zip(axes, methods):

        components = np.abs(results[num_obs][method]["a_est"])
        ax.scatter(range(len(components)), components, s=10, color='skyblue')

        ax.set_xlabel("Rank")
        ax.set_ylabel("Principal Component")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title(f"{method} Principal Component")

    plt.suptitle(f"Principal Component - Patch {patch_idx}")
    plt.tight_layout()
    plt.savefig(f"pc_per_method_patch_{patch_idx}.svg", dpi=300)
    plt.close()

'''
Plot squared error
'''

 
def plot_smoothed_error(ax, err, label):
    df = pd.DataFrame({"Index": range(len(err)), "Error": err})
    # rolling mean (window of 15 components) to smooth the curve
    df["Smoothed_Error"] = df["Error"].rolling(15, min_periods=1).mean()

    ax.plot(df["Index"], df["Smoothed_Error"], label=label)

def plot_cdf_error(ax, err, label):
    err = np.array(err)
    sorted_err = np.sort(err) # TODO: instead of sort - do the cumulative sum of the error
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax.plot(sorted_err, cdf, label=label)

def cumsum_err(ax, err, label):
    err = np.array(err)
    cumsum_err = np.cumsum(err)
    x = np.arange(1, len(err) + 1)
    ax.plot(x, cumsum_err, label=label)

def plot_ranked_errors(ax, err, label):
    err = np.array(err)
    sorted_err = np.sort(err)  # ascending order
    ranks = np.arange(1, len(err) + 1)
    ax.plot(ranks, sorted_err, label=label)


def compare_smoothed_errors(results, num_obs_list, filename, patch_idx):
    n_obs = len(num_obs_list)
    fig, axes = plt.subplots(1, n_obs, figsize=(8*n_obs, 6))  # width scales with number of plots

    # if only 1 subplot, axes is not a list, make it a list
    if n_obs == 1:
        axes = [axes]

    for ax, num_obs in zip(axes, num_obs_list):
        for method in ["V1", "Pixel", "Gaussian"]:
            cumsum_err(ax, results[num_obs][method]["error"], method)

        ax.set_xscale('linear')
        ax.set_yscale('linear')
        ax.set_title(f"Error per Component - Patch {patch_idx}")
        ax.set_xlabel("Index")
        ax.set_ylabel("Cumulative Squared Error")
        ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

'''
PCs as pics
'''
# top of each method
def plot_first_pc(results, num_obs, cmap="gray", title=None, figsize=(12, 4), fileName=None):
    methods = ["V1", "Pixel", "Gaussian"]
    n_methods = len(methods)
    
    plt.figure(figsize=figsize)
    
    for i, method in enumerate(methods):
        pc_dct = results[num_obs][method]["Vh"][0, :].reshape(32, 32)
        pc = fft.idctn(pc_dct, norm = 'ortho', axes = [0, 1])
        ax = plt.subplot(1, n_methods, i+1)
        ax.imshow(pc, cmap=cmap)
        ax.axis("off")
        ax.set_title(f'{method} First PC', fontsize=12)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(fileName, dpi=300)
    plt.close()

def plot_top_pcs(results, num_obs, num_pcs=3, cmap="gray", title=None, figsize=(12, 8), fileName=None):
    methods = ["V1", "Pixel", "Gaussian"]
    n_methods = len(methods)

    plt.figure(figsize=figsize)

    for row, method in enumerate(methods):
        Vh = results[num_obs][method]["Vh"]

        for col in range(num_pcs):
            pc_dct = Vh[col, :].reshape(PATCH_SIZE, PATCH_SIZE)
            pc = fft.idctn(pc_dct, norm = 'ortho', axes = [0, 1])

            ax = plt.subplot(n_methods, num_pcs, row * num_pcs + col + 1)
            ax.imshow(pc, cmap=cmap)
            ax.axis("off")

            # PC label
            ax.set_title(f"PC {col + 1}", fontsize=10)

            # method label
            if col == 0:
                ax.annotate(method, xy=(-0.25, 0.5), xycoords="axes fraction", rotation=90, ha="right", va="center", fontsize=12)
                
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fileName, dpi=300)
    plt.close()

'''
Sparcity of coeffs vectors - histogram of entries in coeffs vectors
'''
def coeff_vectors_hist(results, num_obs, patch_idx):
    plt.figure(figsize=(16, 4))

    # labels and coeffs
    coeff_vectors = [
        ("V1 Estimated", results[num_obs]["V1"]["est_coeffs"].flatten()),
        ("Pixel Estimated", results[num_obs]["Pixel"]["est_coeffs"].flatten()),
        ("Gaussian Estimated", results[num_obs]["Gaussian"]["est_coeffs"].flatten()),
        ("True", results[num_obs]["coeffs_true"].flatten()),
    ]
    
    all_abs = np.concatenate([np.abs(c) for _, c in coeff_vectors])
    upper = np.percentile(all_abs, 99)   # 99th percentile
    bins = np.linspace(0, upper, 50)


    for i, (label, coeffs) in enumerate(coeff_vectors):
        ax = plt.subplot(1, 4, i + 1)
        ax.hist(np.abs(coeffs), bins=bins, edgecolor="black", color='C'+str(i))
        ax.set_xlabel("Coefficient Magnitude")
        ax.set_ylabel("Number of Coefficients")
        ax.set_title(label)
        #ax.set_ylim(0, 12)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)

    plt.suptitle(f"Coefficient Histograms - Patch {patch_idx}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"coeff_histograms_{num_obs}_patch_{patch_idx}.svg", dpi=300)
    plt.close()

    print(f"\nNumber of coefficients <0.1 and <0.5 ({num_obs} Obs):")
    for label, coeffs in coeff_vectors:
        less_than_01 = np.sum(np.abs(coeffs) < 0.1)
        less_than_05 = np.sum(np.abs(coeffs) < 0.5)
        print(f"{label:15s}  <0.1: {less_than_01:4d},  <0.5: {less_than_05:4d}")

# cdf of coeffs
def coeff_vectors_cdf(results, num_obs, patch_idx):
    plt.figure(figsize=(6, 5))

    # labels, coefficient vectors
    coeff_vectors = [
        ("True", results[num_obs]["coeffs_true"].flatten()),
        ("V1 Estimated", results[num_obs]["V1"]["est_coeffs"].flatten()),
        ("Pixel Estimated", results[num_obs]["Pixel"]["est_coeffs"].flatten()),
        ("Gaussian Estimated", results[num_obs]["Gaussian"]["est_coeffs"].flatten()),
    ]

    for label, coeffs in coeff_vectors:
        abs_coeffs = np.sort(np.abs(coeffs))
        cdf = np.arange(1, len(abs_coeffs) + 1) / len(abs_coeffs)
        plt.plot(abs_coeffs, cdf, label=label)

    plt.xscale("log")
    plt.xlabel("Absolute Coefficient Value")
    plt.ylabel("CDF")
    plt.title(f"CDF of Coefficients - Patch {patch_idx}")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"coeff_cdf_{num_obs}_patch_{patch_idx}.svg", dpi=300)
    plt.close()

for patch_idx, patch_results in results.items():
    results = {256: patch_results}
    pc_per_method(results, 256, patch_idx)
    # pc_scatter_plots(results, 256, f"PC_scatter_patch_{patch_idx}.svg", patch_idx)
    # compare_smoothed_errors(results, [256], f"smoothed_error_cdf_patch_{patch_idx}.svg", patch_idx)
    # plot_top_pcs(results, num_obs=256, num_pcs=3,
    #                 title=f"Principal Components per Method  - Patch {patch_idx}",
    #                 fileName=f"pc_top3_images_256_patch_{patch_idx}.png", 
    # )
    # coeff_vectors_hist(results, 256, patch_idx)
    # coeff_vectors_cdf(results, 256, patch_idx)

def pc_scatter_plots_all_patches(results, patch_idxs, filename, cmap='cool'):

    n_rows = len(patch_idxs)
    methods = ["V1", "Pixel", "Gaussian"]

    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows), sharey=True)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # get the global max and min for y=x line
    global_min = np.inf
    global_max = -np.inf

    for patch_idx in patch_idxs:
        for method in methods:
            est = np.abs(results[patch_idx][method]["a_est"])
            true = np.abs(results[patch_idx][method]["a_true"])

            combined = np.concatenate([est, true])
            combined = combined[combined > 0]

            global_min = min(global_min, combined.min())
            global_max = max(global_max, combined.max())

    for row, patch_idx in enumerate(patch_idxs):

        for col, method in enumerate(methods):

            ax = axes[row, col]

            est = results[patch_idx][method]["a_est"]
            true = results[patch_idx][method]["a_true"]

            sc = ax.scatter(
                np.abs(est),
                np.abs(true),
                c=np.arange(len(est)),
                s=30,
                cmap=cmap,
                alpha=0.5
            )

            ax.set_xscale('log')
            ax.set_yscale('log')

            ax.set_xlim(global_min, global_max)
            ax.set_ylim(global_min, global_max)

            ax.plot([global_min, global_max], [global_min, global_max], '--', color='gray')

            ax.set_aspect('equal', adjustable='box')

            if row == 0:
                ax.set_title(method, fontsize=18)
            if col == 0:
                ax.set_ylabel(f"Patch {patch_idx}\n\nTrue PC", fontsize=18)
            if row == n_rows - 1: 
                ax.set_xlabel("Estimated PC", fontsize=18)
            if col != 0:
                ax.yaxis.set_visible(False)
            if col == 2:
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.01)
                cbar = plt.colorbar(sc, cax=cax)
                cbar.set_label("PC rank", rotation=270, labelpad=15)

    fig.suptitle("True vs Estimated Principal Components", fontsize=20, x=0.53)
    plt.tight_layout(rect=[0, 0, 0.92, 0.97])
    plt.savefig(filename, format="svg")
    plt.close()

def pc_per_method_all_patches(results, patch_idxs, filename):

    n_rows = len(patch_idxs)
    methods = ["V1", "Pixel", "Gaussian"]

    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows), sharey=True)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, patch_idx in enumerate(patch_idxs):

        for col, method in enumerate(methods):
            ax = axes[row, col]

            components = np.cumsum((results[patch_idx][method]["a_est"]))
            # ranks = np.arange(1, len(components) + 1)
            # ax.scatter(ranks, components, s=10, color='skyblue')
            ax.plot(range(len(components)), components, s=10, color='skyblue')

            ax.set_yscale('log')
            #ax.set_xscale('log')
            if row == 0:
                ax.set_title(method, fontsize=18)
            if col == 0:
                # TODO: change label to be images something PC
                ax.set_ylabel(f"Patch {patch_idx}\n\n PC", fontsize=18)
            if row == n_rows - 1: 
                ax.set_xlabel("Rank", fontsize=18)
            if col != 0:
                ax.yaxis.set_visible(False)

    fig.suptitle("Estimated Principal Components", fontsize=20, x=0.55)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename, format="svg")
    plt.close()

def error_all_patches(results, patch_idxs, filename):
    n_rows = (len(patch_idxs) + 1) // 2
    n_cols = 2
    methods = ["V1", "Pixel", "Gaussian"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows) )
    axes = axes.flatten()
    
    for i, patch_idx in enumerate(patch_idxs):
        ax = axes[i]
        for method in methods:
            #plot_smoothed_error(ax, results[patch_idx][method]["error"], method)
            cumsum_err(ax, results[patch_idx][method]["error"], method)
            print(patch_idx, method, np.cumsum(results[patch_idx][method]["error"]))
        #print()
        # for cumsum:
        ax.set_xscale('linear')
        ax.set_yscale('linear')

        # for ranked:
        # ax.set_xscale("linear")
        # ax.set_yscale("log")

        # for cdf: 
        # ax.set_xscale("log")
        # ax.set_yscale("linear")

        # for rolling mean: 
        # ax.set_xscale("log")
        # ax.set_yscale("log")

        ax.set_title(f"Patch {patch_idx}", fontsize=16)

        # only set xlabel for bottom row
        row = i // n_cols
        col = i % n_cols
        if row == n_rows - 1:
            ax.set_xlabel("Index", fontsize=14)
        if col == 0:
            ax.set_ylabel("Cumulative Squared Error", fontsize=14)
        # if col != 0: 
        #     ax.yaxis.set_visible(False)
        # ymin, ymax = ax.get_ylim()
        #print(patch_idx, np.sum(results[patch_idx]["V1"]["error"]))
        # print(f"Patch: {patch_idx}, Y-axis limits: min={ymin}, max={ymax}")

        ax.legend()

    # remove extra empty axes
    for ax in axes[len(patch_idxs):]:
        fig.delaxes(ax)

    fig.suptitle("Error per Component", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, format="svg")
    plt.close()

def coeff_vectors_hist_all_patches(results, patch_idxs, filename):

    # TODO: swap order so true is last so that the color order is right
    n_rows = len(patch_idxs)
    n_cols = 4 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), sharey=True)
    
    # flatten axes if 1 row
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, patch_idx in enumerate(patch_idxs):
        coeff_vectors = [
            ("True", results[patch_idx]["coeffs_true"].flatten()),
            ("V1 Estimated", results[patch_idx]["V1"]["est_coeffs"].flatten()),
            ("Pixel Estimated", results[patch_idx]["Pixel"]["est_coeffs"].flatten()),
            ("Gaussian Estimated", results[patch_idx]["Gaussian"]["est_coeffs"].flatten()),
        ]
        all_abs = np.concatenate([np.abs(c) for _, c in coeff_vectors])
        upper = np.percentile(all_abs, 99)   # 99th percentile
        bins = np.linspace(0, upper, 50)

        for col, (label, coeffs) in enumerate(coeff_vectors):
            ax = axes[row, col]
            ax.hist(np.abs(coeffs), bins=bins, edgecolor="black", color=f"C{col}")
            #ax.set_ylim(0, 12)
            ax.set_yscale("log")
            ax.grid(True, which='major', linestyle='--', alpha=0.4)

            if row == 0:
                ax.set_title(label, fontsize=12)
            if col == 0:
                ax.set_ylabel(f"Patch {patch_idx}\n\nCount", fontsize=12)
            if row == n_rows - 1:
                ax.set_xlabel("Coefficient Magnitude", fontsize=12)
            if col != 0: 
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    plt.suptitle(f"Frequency of Coefficients", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, format="svg", dpi=300)
    plt.close()

def coeff_vectors_cdf_all_patches(results, patch_idxs, filename):
    n_rows = (len(patch_idxs) + 1) // 2
    n_cols = 2
    methods = ["V1", "Pixel", "Gaussian"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten()
    
    for i, patch_idx in enumerate(patch_idxs):
        ax = axes[i]
        coeff_vectors = [
            ("True", results[patch_idx]["coeffs_true"].flatten()),
            ("V1 Estimated", results[patch_idx]["V1"]["est_coeffs"].flatten()),
            ("Pixel Estimated", results[patch_idx]["Pixel"]["est_coeffs"].flatten()),
            ("Gaussian Estimated", results[patch_idx]["Gaussian"]["est_coeffs"].flatten()),
        ]

        for label, coeffs in coeff_vectors:
            abs_coeffs = np.sort(np.abs(coeffs))
            cdf = np.arange(1, len(abs_coeffs) + 1) / len(abs_coeffs)
            ax.plot(abs_coeffs, cdf, label=label)

        ax.set_xscale('log')
        #ax.set_yscale('log')
        ax.set_title(f"Patch {patch_idx}", fontsize=16)

        row = i // n_cols
        col = i % n_cols
        if row == n_rows - 1:
            ax.set_xlabel("Absolute Coefficient Value", fontsize=14)
        if col == 0:
            ax.set_ylabel("CDF", fontsize=14)
        if col != 0: 
            ax.yaxis.set_visible(False)
        ax.legend()

    # remove extra empty axes
    for ax in axes[len(patch_idxs):]:
        fig.delaxes(ax)

    fig.suptitle("CDF of Coefficients", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, format="svg")
    plt.close()


# pc_scatter_plots_all_patches(results,PATCH_IDXS,"all_patches_pc_scatter.svg")
# pc_per_method_all_patches(results, PATCH_IDXS,"all_patches_pc_per_method_cumsum.svg")
#error_all_patches(results, PATCH_IDXS, "all_patches_error_cumsum.svg")
# coeff_vectors_hist_all_patches(results, PATCH_IDXS, "all_patches_coeffs_hist_full_y.svg" )
# coeff_vectors_cdf_all_patches(results, PATCH_IDXS, "all_patches_coeffs_cdf_lim.svg")