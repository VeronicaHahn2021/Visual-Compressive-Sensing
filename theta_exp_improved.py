import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.compress_sensing import *
from src.utility import *
from PIL import Image, ImageOps

small_img = "tree_part1.jpg"
big_img="peppers.png"
method = 'dct'
observation="pixel"
mode = '-c'
alpha=0.1
num_cell_100 = 100
num_cell_300 = 300
cell_size = 5    # receptive field size
sparse_freq = 1  # blob size

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

#V1_W_100, V1_y_100 = generate_V1_observation(small_img_arr_gray, num_cell_100, cell_size, sparse_freq)
#V1_W_300, V1_y_300 = generate_V1_observation(small_img_arr_gray, num_cell_300, cell_size, sparse_freq)

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

def dot_product_matrix(img_arr, observation):
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
        W, Y = generate_V1_observation(img_arr, num_cell_300, cell_size, sparse_freq)
        theta = generate_theta(W)
    if observation == "pixel":
        W, Y = generate_pixel_observation(img_arr, num_cell_300)
        theta = generate_theta(W)
    if observation == "gaussian":
        W, Y = generate_gaussian_observation(img_arr, num_cell_300)
        theta = generate_theta(W)

    col_norms = np.linalg.norm(theta, axis=0)
    x = theta / col_norms 
    M = x.T @ x
    np.fill_diagonal(M, 0)
    return np.abs(M)

def mutual_coherence_matrix(A, n, num_cell, observation, sparse_freq = None) :
    '''
    Create a list of n computed mutual coherence(MC) values for given observations A

    A: array_like(?)
        Image // ex) small_img

    n: int 
        how many MC should be collected from one image, 
        with purpose of averaging and comparing

    observation: str
        'V1', 'pixel', 'gaussian', which observation type we're computing for 

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
            W, Y = generate_V1_observation(A, num_cell, cell_size, sparse_freq)
            theta = generate_theta(W)
            M[i] = compute_mutual_coherence(sort_theta(theta))
            #M[i] = compute_mutual_coherence((theta))
        if observation == "pixel":
            W, Y = generate_pixel_observation(A, num_cell)
            theta = generate_theta(W)
            M[i] = compute_mutual_coherence(theta)
        if observation == "gaussian":
            W, Y = generate_gaussian_observation(A, num_cell)
            theta = generate_theta(W)
            M[i] = compute_mutual_coherence(theta)
    return M

'''
#Plot Mutual Coherence - WORKING for small_gray
num = 3
v1_mc = mutual_coherence_matrix(small_img_arr_gray, num, num_cell_300,  "V1", sparse_freq)
pix_mc = mutual_coherence_matrix(small_img_arr_gray, num, num_cell_300, "pixel")
gaus_mc = mutual_coherence_matrix(small_img_arr_gray, num,num_cell_300, "gaussian")
all_mc = [v1_mc, pix_mc, gaus_mc]
fig = plt.figure()
fig.suptitle("Average Mutual Coherence", fontsize=14)
ax = fig.add_subplot()
ax.boxplot(all_mc, tick_labels=['V1', 'pixel','Gaussian'])
ax.set_xlabel("V1, Pix, Gaus")
plt.show()
'''

def sort_theta(theta):
    arr = np.arange(30)
    kx = np.tile(arr, 30)
    ky = np.repeat(arr, 30)

    ksum = kx**2 + ky**2
    perm = np.argsort(ksum) 
    return theta[:, perm]


#Plot Dot Products - WORKING for small_gray
v1_dot = dot_product_matrix(small_img_arr_gray, "V1")
v1_upper_dot = np.triu(v1_dot, k=1)
pix_dot = dot_product_matrix(small_img_arr_gray, "pixel")
pix_upper_dot = np.triu(pix_dot, k=1)
gaus_dot = dot_product_matrix(small_img_arr_gray, "gaussian")
gaus_upper_dot = np.triu(gaus_dot, k=1)

bins = np.linspace(0,0.35, 50)
plt.figure()
plt.imshow(v1_dot, interpolation=None)
plt.colorbar()


plt.figure()

plt.hist(v1_dot.flatten(), bins, cumulative=False, density=True, label='v1')
plt.xlabel('Dot Product')
plt.ylabel('Frequency')
plt.title('V1 Dot Products')
# plt.show()

# plt.figure()
plt.hist(pix_dot.flatten(), bins, cumulative=False, density=True, label='pix')
plt.xlabel('Dot Product')
plt.ylabel('Frequency')
plt.title('Pixel Dot Products')
#plt.show()

#plt.figure()
plt.hist(gaus_dot.flatten(), bins, cumulative=False, density=True, label='gauss')
plt.xlabel('Dot Product')
plt.ylabel('Frequency')
plt.title('Gaussian Dot Products')
plt.legend()

plt.yscale('log')

plt.show()


