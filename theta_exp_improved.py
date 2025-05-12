import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.compress_sensing import *
from src.utility import *
from PIL import Image, ImageOps
import sys

small_img = "tree_part1.jpg"
big_img="peppers.png"
method = 'dct'
observation="pixel"
mode = '-c'
alpha=0.1
num_cell_100 = 100
num_cell_300 = 300
cell_size = 200    # receptive field size 
sparse_freq = .001  # blob size            
num = 20

## For wavelet variable
lv= 2
dwt_type= 'db2'


plt.ion()

#Load Images:
# Represent image as numpy array to make it easier to process
small_img_arr = process_image(small_img, mode)
small_img_arr_gray = process_image(small_img, False) #change from 'gray' to False
big_img_arr = process_image(big_img, mode)
big_img_arr_gray = process_image(big_img, False) #change from 'gray' to False


def generate_theta(W):
    '''
    Generate theta for given weight matrix

    Parameters
    ----------

    W: array_like
        Lists of weighted data
    '''

    num_cell, n, m = W.shape
    theta = fft.dctn(W.reshape(num_cell, n, m), norm = 'ortho', axes = [1, 2])
    theta = theta.reshape(num_cell, n * m) # PASS INTO COHERENCE FUNCTION 
    return theta

def compute_mutual_coherence(theta) :
    '''
    Compute mutual coherence for generic given matrix

    Parameters
    ----------

    A: array_like
        matrix with more than one column

    The how:
    1. normalize columns of A (divide each by its norm):
       collect n = columns, m = rows
       for each column n, compute col_norm = sqrt(a_n1^2 + a_n2^2 ... + a_nm^2)
            for each a in n, a = a/col_norm
       A is now an array of normalized columns
    2. find max dot product between columns of A = mutual coherence
        create array total_dot
        for each column x in A
            create array dot = dot products between col x with every column after it
            add dot to total_dot
        return max(total_dot)

    '''
    col_norms = np.linalg.norm(theta, axis=0)
    x = theta / col_norms 
    M = x.T @ x 
    np.fill_diagonal(M, 0) 
    return np.abs(M).flatten().max() 

def dot_product_matrix(img_arr, observation, num_cell, cell_size = None, sparse_freq = None):
    '''
    Create an array of dot products between columns

    Parameters
    ----------

    img_arr: numpy_array
        (n, m) shape image containing array of pixels.

    observation: String
        Observation technique that are going to be used to 
        collect sample for reconstruction. Default set up to 'pixel'
        Supported observation : ['pixel', 'gaussian', 'V1'].
    '''

    if observation == 'V1':
        W, Y = generate_V1_observation(img_arr, num_cell, cell_size, sparse_freq)
        theta = generate_theta(W)
    if observation == "pixel":
        W, Y = generate_pixel_observation(img_arr, num_cell)
        theta = generate_theta(W)
    if observation == "gaussian":
        W, Y = generate_gaussian_observation(img_arr, num_cell)
        theta = generate_theta(W)

    col_norms = np.linalg.norm(theta, axis=0)
    x = theta / col_norms 
    M = x.T @ x
    np.fill_diagonal(M, 0)
    return np.abs(M)

def mutual_coherence_matrix(img_arr, n, num_cell, observation, cell_size = None, sparse_freq = None) :
    '''
    Returns a list of n computed mutual coherence(MC) values for given image and observation type

    img_arr: array_like
        I(n, m) shape image containing array of pixels

    n: int 
        how many MC should be collected from one image, 
        with purpose of averaging and comparing

    observation: String
        Observation technique that are going to be used to 
        collect sample for reconstruction. Default set up to 'pixel'
        Supported observation : ['pixel', 'gaussian', 'V1']. 
    
    

    The how:
    1. Create array M, will be our final list of MCs
    2. for n times, generate theta and compute mutual coherence depending on ovserbation type
        add each MC value to M
    3. return M - to be plotted
    
    '''

    M = np.zeros(n)
    i = 0
    for i in range(n):
        if observation == 'V1':
            W, Y = generate_V1_observation(img_arr, num_cell, cell_size, sparse_freq)
            theta = generate_theta(W)
            #M[i] = compute_mutual_coherence(sort_theta(theta))
            M[i] = compute_mutual_coherence((theta))
        if observation == "pixel":
            W, Y = generate_pixel_observation(img_arr, num_cell)
            theta = generate_theta(W)
            M[i] = compute_mutual_coherence(theta)
        if observation == "gaussian":
            W, Y = generate_gaussian_observation(img_arr, num_cell)
            theta = generate_theta(W)
            M[i] = compute_mutual_coherence(theta)
    return M

def sort_theta(theta):
    arr = np.arange(30)
    kx = np.tile(arr, 30)
    ky = np.repeat(arr, 30)

    ksum = kx**2 + ky**2
    perm = np.argsort(ksum) 
    return theta[:, perm]


def generate_coeff_vector(img_arr, num_cell, cell_size, sparse_freq):
    '''
    Generates the coeffiecient vector for frequencies present in img
    '''

    n, m = img_arr.shape
    c = fft.dctn(img_arr, norm = 'ortho', axes = [0, 1])    
    return c   #.reshape(n*m,1)

def generate_ctMc(img_arr, obs_type, num_cell, norm = 2, diagonal = 0, cell_size = None, sparse_freq = None):
    '''
    Returns coefficient matrix * dot product matrix based on norm and what 
    values the diagonal is set to

    img_arr: array_like
        I(n, m) shape image containing array of pixels

    observation: String
        Observation technique that are going to be used to 
        collect sample for reconstruction. Default set up to 'pixel'
        Supported observation : ['pixel', 'gaussian', 'V1'].

    num_cell : int
        Number of blobs that will be used to be 
        determining which pixels to grab and use.
    
    norm: int
        np.linalg.norm(coeffs) ** norm
        norm type to divide by

    diagonal: int
        Number to replace diagonal values with in dot vector
        0: will return metric without altering diagonal
        diagonal >0: will return metric having replaced dot_vec diagonal

    cell_size : int
        Determines field size of opened and closed blob of data. 
        Affect the data training.
        
    sparse_freq : int
        Determines filed frequency on how frequently 
        opened and closed area would appear. 
        Affect the data training.
    '''
    coeffs= generate_coeff_vector(small_img_arr_gray,num_cell,cell_size,sparse_freq).flatten()
    dot_vec = dot_product_matrix(img_arr, obs_type, num_cell, cell_size, sparse_freq)
    dot_vec = np.linalg.inv(dot_vec)
    
    if diagonal >= 1:
        metric = np.fill_diagonal(dot_vec, diagonal)
        metric = coeffs.T @ dot_vec @ coeffs / np.linalg.norm(coeffs) ** norm
        return metric
    else:
        metric = coeffs.T @ dot_vec @ coeffs / np.linalg.norm(coeffs) ** norm
        return metric
    
def generate_dot_metric(img_arr, obs_type, num_cell, norm = 1, cell_size = None, sparse_freq = None):
    coeffs= generate_coeff_vector(small_img_arr_gray,num_cell,cell_size,sparse_freq).flatten()
    dot_vec = dot_product_matrix(img_arr, obs_type, num_cell, cell_size, sparse_freq)

    if norm <= 0:
        return np.linalg.norm(dot_vec @ coeffs, np.inf)
    else:
        return np.linalg.norm(dot_vec @ coeffs, norm)

#~~~~~~~~~
coeffs= generate_coeff_vector(small_img_arr_gray,num_cell_300,cell_size,sparse_freq).flatten() #c vector

M_pix = dot_product_matrix(small_img_arr_gray, 'pixel', num_cell_300)
# M_pix = np.linalg.inv(M_pix)
    # metricPix = coeffs.T @ M_pix @ coeffs / np.linalg.norm(coeffs) ** 2
    # np.fill_diagonal(M_pix, 1)
# metricPix_diag1 = coeffs.T @ M_pix @ coeffs / np.linalg.norm(coeffs) ** 2
# # Mhalf = np.linalg.cholesky(M_pix, upper=True)
# metricPix_3 = np.linalg.norm(M_pix @ coeffs, 1)
# print(f"Pix: {metricPix}, {metricPix_diag1}, {metricPix_3}")
print(f"Pix: {np.linalg.norm(M_pix @ coeffs, np.inf)}")

M_gaus = dot_product_matrix(small_img_arr_gray, 'gaussian', num_cell_300)
# # M_gaus = np.linalg.inv(M_gaus)
# metricGaus = coeffs.T @ M_gaus @ coeffs / np.linalg.norm(coeffs) ** 2
# np.fill_diagonal(M_gaus, 1)
# metricGaus_diag1 = coeffs.T @ M_gaus @ coeffs / np.linalg.norm(coeffs) ** 2
# metricGaus_3 = np.linalg.norm(M_gaus @ coeffs, 1)
# print(f"Gaus: {metricGaus}, {metricGaus_diag1}, {metricGaus_3}")
print(f"Gaussian: {np.linalg.norm(M_gaus @ coeffs, np.inf)}")

M_V1 = dot_product_matrix(small_img_arr_gray, 'V1', num_cell_300, cell_size, sparse_freq)
# # M_V1 = np.linalg.inv(M_V1)
# metricV1 = coeffs.T @ M_V1 @ coeffs / np.linalg.norm(coeffs) ** 2 
# np.fill_diagonal(M_V1, 1)
# metricV1_diag1 = coeffs.T @ M_V1 @ coeffs / np.linalg.norm(coeffs) ** 2
# metricV1_3 = np.linalg.norm(M_V1 @ coeffs, 1)
# print(f"V1: {metricV1}, {metricV1_diag1}, {metricV1_3}")
print(f"V1: {np.linalg.norm(M_V1 @ coeffs, np.inf)}")

sys.exit()

M_vec = np.ravel(M) # dot product matrix as a vector
#M_coords will be the coordinates of each coherence in M, which
#we want to keep track of bc that'll tell us where the largest coherences are
n = M.shape[0]
M_coords = np.unravel_index(range(n**2), (n,n))
arr = np.arange(30)
kx = np.tile(arr, 30)
ky = np.repeat(arr, 30)
perm = np.argsort(M_vec)[::-1] # tells how to sort, reversed to decrease
M_vec[perm] # decreasing
i = M_coords[0][perm]
j = M_coords[1][perm]
# look into kx, ky to map i -> (kx, ky)
i_coords=[[]for s in range(i.shape[0])]
j_coords=[[]for s in range(j.shape[0])]
for m in range(i.shape[0]):
    i_coords[m] = [kx[i[m]],ky[i[m]]] #assigns kx,ky coordinates to elements in i
for m in range(j.shape[0]):
    j_coords[m] = [kx[j[m]],ky[j[m]]] #assigns kx,ky coordinates to elements in j
i_arr = [i,i_coords]
j_arr = [j,j_coords]
i_sorted = pd.DataFrame({
    "i": i,
    "(kx,ky)": i_coords,
    "MC": M_vec[perm]
})
j_sorted = pd.DataFrame({
    "j": j,
    "(kx,ky)": j_coords,
    "MC": M_vec[perm]
})

coeff_vec = generate_coeff_vector(small_img_arr_gray,num_cell_300,cell_size,sparse_freq) #c vector
