import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.compress_sensing import *
from src.utility import *
from PIL import Image, ImageOps

#Preparing parameters needed for the examples:

#Hyperparameter Values
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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))

small_img_arr = process_image(small_img, mode)
ax1.imshow(small_img_arr)
ax1.set_title("{img}".format(img = small_img.split('.')[0]))
ax1.axis('off')
#print(small_img_arr.shape)
small_img_arr_gray = process_image(small_img, False) #change from 'gray' to False
ax2.imshow(small_img_arr_gray, 'gray')
#print(small_img_arr_gray.shape)
ax2.set_title("{img} grayscaled".format(img = small_img.split('.')[0]))
ax2.axis('off')
#plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
big_img_arr = process_image(big_img, mode)
ax1.imshow(big_img_arr)
ax1.set_title("{img}".format(img = big_img.split('.')[0]))
ax1.axis('off')
#print(big_img_arr.shape)
big_img_arr_gray = process_image(big_img, False) #change from 'gray' to False
ax2.imshow(big_img_arr_gray, 'gray')
ax2.set_title("{img} grayscaled".format(img = big_img.split('.')[0]))
ax2.axis('off')
#print(big_img_arr_gray.shape)
#plt.show()

#Collect v1 observations from small image:
V1_W_100, V1_y_100 = generate_V1_observation(small_img_arr_gray, num_cell_100, cell_size, sparse_freq)
V1_W_300, V1_y_300 = generate_V1_observation(small_img_arr_gray, num_cell_300, cell_size, sparse_freq)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))

fig.suptitle("V1 Grascaled Reconstruction")
## Reconstruction with 100 number of cells grayscaled
reconst_gray_100 = reconstruct(V1_W_100, V1_y_100, alpha)
ax1.imshow(reconst_gray_100, 'gray')
ax1.set_title("{num_cell} number of cells".format(num_cell = num_cell_100))
ax1.axis("off")

# ## Reconstruction with 300 number of cells grayscaled
reconst_gray_300 = reconstruct(V1_W_300, V1_y_300, alpha)
ax2.imshow(reconst_gray_300, 'gray')
ax2.set_title("{num_cell} number of cells".format(num_cell = num_cell_300))
ax2.axis("off")
plt.subplots_adjust(top=1.35)
#plt.show()

#trying to print theta/figure out what i need to access it specifically
#turn this inot a function so i can use it on gaussian vs v1 vs pixel?
W = V1_W_100
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
    #print(theta.shape)
    theta = theta.reshape(num_cell, n * m) # PASS INTO COHERENCE FUNCTION
    print(theta.shape) 

generate_theta(W)


test = [[1,2,3], [4,5,6], [7,8,9]]
print(test)


def compute_mutual_coherence(A) :
    '''
    Compute mutual coherence for generic given matrix

    Parameters
    ----------

    A: 2d matrix with more than one columns

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
    col_norms = np.linalg.norm(A, axis=0) # col_norms is an array of norms per column (the 0th axis of A is the row?)
    #print(col_norms)
    x = A / col_norms #x is an array the same shape as A, but all columns have been normalized
    #print(x)
    M = x.T @ x #M is the matrix multiplication between x and itself, diagonals are 1 because columns have been normalized (x*x = 1)
                 #M is a matrix of dot products between each column
    #print(M)
    np.fill_diagonal(M, 0) #replace diagonals with 0 so they don't screw up the max
    #print(M)

    print(np.abs(M).flatten().max())
    return np.abs(M).flatten().max() #max of M

#compute_mutual_coherence(theta)
#plt.close() if stuff wont close, theta[:, x] for column x, imshow to bring up heatmaps of W
#W[x,:,:] is the xth 30x30 measurement, colorbar to bring up the colorbar, x from 1-100
#theta is W gone through the fourier transform, 