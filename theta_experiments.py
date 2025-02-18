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
small_img_arr = process_image(small_img, mode)
small_img_arr_gray = process_image(small_img, False) #change from 'gray' to False
big_img_arr = process_image(big_img, mode)
big_img_arr_gray = process_image(big_img, False) #change from 'gray' to False


#Collect v1 observations from small image:
V1_W_100, V1_y_100 = generate_V1_observation(small_img_arr_gray, num_cell_100, cell_size, sparse_freq)
V1_W_300, V1_y_300 = generate_V1_observation(small_img_arr_gray, num_cell_300, cell_size, sparse_freq)



#trying to print theta/figure out what i need to access it specifically
#turn this into a function so i can use it on gaussian vs v1 vs pixel?
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
    theta = theta.reshape(num_cell, n * m) # PASS INTO COHERENCE FUNCTION 
    return theta

#theta = generate_theta(W)
#print(theta.shape)


def compute_mutual_coherence(A) :
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
    col_norms = np.linalg.norm(A, axis=0) # col_norms is an array of norms per column (the 0th axis of A is the row?)
    x = A / col_norms #x is an array the same shape as A, but all columns have been normalized
    M = x.T @ x #M is the matrix multiplication between x and itself, diagonals are 1 because columns have been normalized (x*x = 1)
                 #M is a matrix of dot products between each column
    np.fill_diagonal(M, 0) #replace diagonals with 0 so they don't screw up the max

    #print(np.abs(M).flatten().max())
    return np.abs(M).flatten().max() #max of M

#compute_mutual_coherence(test)
#plt.close() if stuff wont close, theta[:, x] for column x, imshow to bring up heatmaps of W
#W[x,:,:] is the xth 30x30 measurement, colorbar to bring up the colorbar, x from 1-100
#theta is W gone through the fourier transform, 

#Idea: find average mutual coherence over multiple V1, gaussian, and pixel observations (since
#they're randomized), compare

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

def average_mutual_coherence(A, n, obs_type) :
    '''
    Create an array of average mutual coherences given image A, number of averages to create n, and observation type

    A: array_like(?)
        Image // ex) small_img

    n: int 
        how many ave MC should be collected from one image, 
        over that many runs

    obs_type: str
        'V1', 'pixel', 'gaussian', which observation type we're computing for
    '''

    ave = np.zeros(n)
    for i in range(n):
        M = mutual_coherence_matrix(A, n, obs_type)
        ave[i] = np.average(M)
    return ave


num=3
#plot num amount of average mutual coherences over num runs

ave_v1 = average_mutual_coherence(small_img_arr_gray, num, 'V1')
ave_pix = average_mutual_coherence(small_img_arr_gray, num, 'pixel')
ave_gau = average_mutual_coherence(small_img_arr_gray, num, 'gaussian')
plt.plot(ave_v1, label = 'V1', marker = 'o')
plt.plot(ave_pix, label = "Pixel", marker = 'o')
plt.plot(ave_gau, label = "Gaussian", marker = 'o')
plt.title('Average Mutual Coherence')
plt.ylabel('Ave MC')
plt.legend()
plt.show()
