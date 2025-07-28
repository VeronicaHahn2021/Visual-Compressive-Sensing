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

def high_freq_table(img_arr, obs_type, num_cell, cell_size = None, blob_size = None, center = None):
    '''
    Creates a table identifying which DCT basis frequencies are where

    Parameters
    ----------

    img_arr:
        (n, m) shape image containing array of pixels

    obs_type: String
        Observation technique that are going to be used to 
        collect sample for reconstruction. Default set up to 'pixel'
        Supported observation : ['pixel', 'gaussian', 'V1'].

    num_cell : int
        Number of blobs that will be used to be 
        determining which pixels to grab and use.

    cell_size : int
        Determines field size of opened and closed blob of data. 
        Affect the data training.
        
    blob_size : int
        Determines filed frequency on how frequently 
        opened and closed area would appear. 
        Affect the data training.
    '''

    M = dot_vec = dot_product_matrix(img_arr, obs_type, num_cell, cell_size, blob_size, center)
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
    df = pd.DataFrame({
        "i": i,
        "j": j,
        "(kx_i,ky_i)": i_coords,
        "(kx_j,ky_j)": j_coords,
        "MC": M_vec[perm]
    })
    return df

    df = high_freq_table(small_img_arr_gray, "V1", num_cell_300, cell_size, blob_size)
    display(df)
    print("Hello world")