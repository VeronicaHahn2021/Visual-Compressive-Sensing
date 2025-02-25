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
cell_size = 3
sparse_freq = 1

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

V1_W_100, V1_y_100 = generate_V1_observation(small_img_arr_gray, num_cell_100, cell_size, sparse_freq)
V1_W_300, V1_y_300 = generate_V1_observation(small_img_arr_gray, num_cell_300, cell_size, sparse_freq)

def generate_theta(W):
    '''
    Generate theta for given weight matrix

    Parameters
    ----------

    W: array_like
        Lists of weighted data(?)
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

def dot_product_matrix(img_arr, obs_type):
    '''
    Create an array of dot products between columns
    '''

    if obs_type == 'V1':
        W, Y = generate_V1_observation(img_arr, num_cell_100, cell_size, sparse_freq)
        theta = generate_theta(W)
    if obs_type == "pixel":
        W, Y = generate_pixel_observation(img_arr, num_cell_100)
        theta = generate_theta(W)
    if obs_type == "gaussian":
        W, Y = generate_gaussian_observation(img_arr, num_cell_100)
        theta = generate_theta(W)

    col_norms = np.linalg.norm(theta, axis=0)
    x = theta / col_norms 
    M = x.T @ x
    np.fill_diagonal(M, 0)
    return np.abs(M)

def mutual_coherence_matrix(A, n, obs_type) :
    '''
    Create a list of n computed mutual coherence(MC) values for given observations A

    A: array_like(?)
        Image // ex) small_img

    n: int 
        how many MC should be collected from one image, 
        with purpose of averaging and comparing

    obs_type: str
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
        if obs_type == 'V1':
            W, Y = generate_V1_observation(A, num_cell_100, cell_size, sparse_freq)
            theta = generate_theta(W)
            M[i] = compute_mutual_coherence(theta)
        if obs_type == "pixel":
            W, Y = generate_pixel_observation(A, num_cell_100)
            theta = generate_theta(W)
            M[i] = compute_mutual_coherence(theta)
        if obs_type == "gaussian":
            W, Y = generate_gaussian_observation(A, num_cell_100)
            theta = generate_theta(W)
            M[i] = compute_mutual_coherence(theta)
    return M


#Plot Mutual Coherence - WORKING for small_gray
num = 50
v1_mc = mutual_coherence_matrix(small_img_arr_gray, num, "V1")
pix_mc = mutual_coherence_matrix(small_img_arr_gray, num, "pixel")
gaus_mc = mutual_coherence_matrix(small_img_arr_gray, num, "gaussian")
all_mc = [v1_mc, pix_mc, gaus_mc]
fig = plt.figure()
fig.suptitle("Average Mutual Coherence", fontsize=14)
ax = fig.add_subplot()
ax.boxplot(all_mc)
ax.set_xlabel("V1, Pix, Gaus")
plt.show()


#Plot Dot Products - WORKING for small_gray
v1_dot = dot_product_matrix(small_img_arr_gray, "V1")
pix_dot = dot_product_matrix(small_img_arr_gray, "pixel")
gaus_dot = dot_product_matrix(small_img_arr_gray, "gaussian")

plt.figure()
plt.hist(v1_dot, bins = 30)
plt.xlabel('Dot Product')
plt.ylabel('Frequency')
plt.title('V1 Dot Products')
plt.show()

plt.figure()
plt.hist(pix_dot, bins = 30)
plt.xlabel('Dot Product')
plt.ylabel('Frequency')
plt.title('Pixel Dot Products')
plt.show()

plt.figure()
plt.hist(gaus_dot, bins = 30)
plt.xlabel('Dot Product')
plt.ylabel('Frequency')
plt.title('Gaussian Dot Products')
plt.show()


'''
v1_mc = mutual_coherence_matrix(small_img_arr, num, "V1")
pix_mc = mutual_coherence_matrix(small_img_arr, num, "pixel")
gaus_mc = mutual_coherence_matrix(small_img_arr, num, "gaussian")
plt.plot(v1_mc,label = 'V1', marker = 'o')
plt.plot(pix_mc, label = 'pixel', marker = 'o')
plt.plot(gaus_mc,label = 'gaussian', marker = 'o')
plt.title('Average MC: Tree, Color')
plt.ylabel('Ave MC')
plt.legend()
plt.show()


v1_mc = mutual_coherence_matrix(big_img_arr, num, "V1")
pix_mc = mutual_coherence_matrix(big_img_arr, num, "pixel")
gaus_mc = mutual_coherence_matrix(big_img_arr, num, "gaussian")
plt.plot(v1_mc,label = 'V1', marker = 'o')
plt.plot(pix_mc, label = 'pixel', marker = 'o')
plt.plot(gaus_mc,label = 'gaussian', marker = 'o')
plt.title('Average MC: Peppers, Color')
plt.ylabel('Ave MC')
plt.legend()
plt.show()

v1_mc = mutual_coherence_matrix(big_img_arr_gray, num, "V1")
pix_mc = mutual_coherence_matrix(big_img_arr_gray, num, "pixel")
gaus_mc = mutual_coherence_matrix(big_img_arr_gray, num, "gaussian")
plt.plot(v1_mc,label = 'V1', marker = 'o')
plt.plot(pix_mc, label = 'pixel', marker = 'o')
plt.plot(gaus_mc,label = 'gaussian', marker = 'o')
plt.title('Average MC: Peppers, Grayscale')
plt.ylabel('Ave MC')
plt.legend()
plt.show()
'''