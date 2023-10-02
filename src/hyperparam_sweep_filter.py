import numpy as np
import numpy.linalg as la
import sys
import matplotlib.pyplot as plt

# Package for importing image representation
from PIL import Image, ImageOps

from src.compress_sensing import *
from src.utility import *
from src.args import parse_sweep_args
import pandas as pd
import itertools
import dask
from dask.distributed import Client, progress
import time
import os.path

import argparse
import pywt



def run_sweep(method, img, observation, color, dwt_type, lv,
              alpha_list, num_cell, cell_size, sparse_freq, fixed_weights):
    ''' 
    Generate a sweep over desired hyperparameters and saves results to a file.
    
    Parameters
    ----------
    method : String
        Method of reconstruction ('dwt' or 'dct')
    
    img : String
        Name of image to reconstruct (e.g. 'tree_part1.jpg')

    observation : String
        Method of observation (e.g. pixel, gaussian, v1)
    
    color : Boolean
        Desired mode to reconstruct image.
        True to reconstruct in RGB, False for grayscaled.

    dwt_type : String
        Type of dwt method to be used.
        See pywt.wavelist() for all possible dwt types.
        
    lv : List of int
        List of one or more integers in [1, 4].
        
    alpha_list : List of float
        Penalty for fitting data onto LASSO function 
        to search for significant coefficents.

    num_cell : List of int
        Number of blobs that will be used to be 
        determining which pixels to grab and use.
    
    cell_size : List of int
        Determines field size of opened and closed blob of data. 
        Affect the data training.

    sparse_freq : List of int
        Determines filed frequency on how frequently 
        opened and closed area would appear. Affect the data training
    '''


    delay_list = []
    rep = np.arange(10)
    image_nm = img.split('.')[0]
    img_arr = process_image(img, color)
    print(fixed_weights)
    # call dask
    client = Client()
    # give non-V1 param search space
    if observation.upper() != 'V1':
        # specify search space for dct and dwt params
        if method.lower() == 'dct':
            search_list = [rep, alpha_list, num_cell]
            search = list(itertools.product(*search_list))
            search_df = pd.DataFrame(search, columns= [ 'rep', 'alp',
                                                        'num_cell'])
            sim_wrapper = lambda rep, alp, num_cell: \
                run_sim_dct(method, observation, color,
                            alp, num_cell, img_arr, fixed_weights)
        elif method.lower() == 'dwt':
            search_list = [rep, dwt_type, lv, alpha_list, num_cell]
            search = list(itertools.product(*search_list))             
            search_df = pd.DataFrame(search, columns= [ 'rep', 'dwt_type', 'lv',
                                                        'alp', 'num_cell'])
            sim_wrapper = lambda rep, dwt_type, lv, alp, num_cell: \
                run_sim_dwt(method, observation, color, dwt_type,
                            lv, alp, num_cell, img_arr, fixed_weights)
    # give v1 param search space
    elif observation.upper() == 'V1':
        # specify search space for dct and dwt params
        if method.lower() == 'dct': 
            search_list = [rep, alpha_list, num_cell, cell_size, sparse_freq]
            search = list(itertools.product(*search_list))
            search_df = pd.DataFrame(search,
                                     columns= ['rep', 'alp', 'num_cell',
                                               'cell_size', 'sparse_freq'])
            sim_wrapper = lambda rep, alp, num_cell, cell_size, sparse_freq: \
                run_sim_V1_dct(method, observation, color, alp,
                               num_cell, cell_size, sparse_freq, img_arr, fixed_weights)
        elif method.lower() == 'dwt':
            search_list = [rep, dwt_type, lv, alpha_list, num_cell, cell_size, sparse_freq]
            search = list(itertools.product(*search_list))             
            search_df = pd.DataFrame(search, columns= [ 'rep', 'dwt_type', 'lv',
                                                        'alp', 'num_cell',
                                                        'cell_size', 'sparse_freq'
                                                       ])
            sim_wrapper = lambda rep, dwt_type, lv, alp, num_cell, cell_size, \
                sparse_freq: run_sim_V1_dwt(method, observation, color,
                                            dwt_type, lv, alp, num_cell,
                                            cell_size, sparse_freq, img_arr, fixed_weights)
        else:
            print(f"The observation {observation} is currently not supported.")
            print(" Please try valid observation type.")
    for p in search_df.values:
        delay = dask.delayed(sim_wrapper)(*p)
        delay_list.append(delay)
    futures = dask.persist(*delay_list)
    progress(futures)
    # Compute the result
    results = dask.compute(*futures, num_workers = 8)
    
    # Saves Computed data to csv file format
    results_df = pd.DataFrame(results, columns=['error'])#, 'theta', 'reform', 's'])
    param_csv_nm = "param_"
    param_path = data_save_path(image_nm, method, observation,
                                f'{color}_{param_csv_nm}')
    # Add error onto parameter
    params_result_df = search_df.join(results_df['error'])
    params_result_df.to_csv(param_path, index=False)
    
    # Saves hyperparameter used for computing this data to txt file format
    hyperparam_track = data_save_path(image_nm, method, observation,
                                      '{color}_hyperparam'.format(color = color))
    f = open(hyperparam_track, 'a+')
    hyperparam_list = list(zip(search_df.columns, search_list))
    f.write(f"{param_path.split('/')[-1]}\n")
    for hyperparam in hyperparam_list :
        f.write(f"   {hyperparam[0]}: {hyperparam[1]}\n")
    f.write("\n\n")
    f.close()
    
    # Terminate Dask properly
    client.close()

# run sim for non-v1 dwt
def run_sim_dwt(method, observation, color, dwt_type,
                lv, alpha, num_cell, img_arr, fixed_weights):
    ''' 
    Run a sim for non-v1 dwt
    
    Parameters
    ----------
    method : String
        Method of reconstruction ('dwt' or 'dct')
    
    observation : String
        Method of observation (e.g. pixel, gaussian, v1)

    color : Boolean
        Desired mode to reconstruct image.
        True to reconstruct in RGB, False for grayscaled.

    dwt_type : String
        Type of dwt method to use.
        See pywt.wavelist() for all possible dwt types.
    
    lv : int
        Generate level of signal frequencies when dwt is used. 
        Should be in [1, 4].

    alpha : float
        Penalty for fitting data onto LASSO function 
        to search for significant coefficents

    num_cell : int
        Number of blobs that will be used to be 
        determining which pixels to grab and use.

    img_arr : numpy_array
        (n, m) shape image containing array of pixels
    
    Returns
    ----------
    error : float
        Computed normalized error value per each pixel
    '''
    dim = img_arr.shape
    if (len(dim) == 3) :
    	n, m, rgb = dim
    else :
    	n, m = dim
    # Deal with fraction num_cell amount
    if (num_cell < 1):
        num_cell = round(n * m * num_cell)
    num_cell = int(num_cell)
    lv = int(lv)
    alpha = float(alpha)
    img_arr = np.array([img_arr]).squeeze()
    reconst = large_img_experiment(img_arr, num_cell = num_cell, alpha = alpha,
                                   method = method, observation = observation,
                                   color = color, lv = lv, dwt_type = dwt_type, 
                                   fixed_weights=fixed_weights)

    # Call function and calculate error
    error = error_calculation(img_arr, reconst)
    print(error)
    
    return error


# run sim for v1 dwt
def run_sim_V1_dwt(method, observation, color, dwt_type,
                   lv, alpha, num_cell, cell_size, sparse_freq, img_arr, fixed_weights):
    ''' 
    Run a sim for v1 dwt
    
    Parameters
    ----------
    method : String
        Method of reconstruction ('dwt' or 'dct')

    observation : String
        Method of observation (e.g. pixel, gaussian, v1)
    
    color : Boolean
        Desired mode to reconstruct image.
        True to reconstruct in RGB, False for grayscaled.

    dwt_type : String
        Type of dwt method to use.
        See pywt.wavelist() for all possible dwt types.

    lv : int
        Generate level of signal frequencies when dwt is used. 
        Should be in [1, 4].

    alpha : float
        Penalty for fitting data onto LASSO function to 
        search for significant coefficents

    num_cell : int
        Number of blobs that will be used to 
        be determining which pixels to grab and use.
    
    cell_size : int
        Determines field size of opened and closed blob of data. 
        Affect the data training

    sparse_freq : int
        Determines filed frequency on how frequently 
        opened and closed area would appear. Affect the data training

    img_arr : numpy_array
        (n, m) shape image containing array of pixels
        
    Returns
    ----------
    error : float
        Computed normalized error value per each pixel
    '''
    dim = img_arr.shape
    if (len(dim) == 3) :
    	n, m, rgb = dim
    else :
    	n, m = dim
    # Deal with fraction num_cell amount
    if (num_cell < 1):
        num_cell = round(n * m * num_cell)
    num_cell = int(num_cell)
    lv = int(lv)
    alpha = float(alpha)
    
    img_arr = np.array([img_arr]).squeeze()
    #Filter reconst to make sure it can reconstruct any size 
    reconst = large_img_experiment(img_arr, num_cell = num_cell,
                                   cell_size = cell_size, sparse_freq = sparse_freq,
                                   alpha = alpha, method = method,
                                   observation = observation, color = color,
                                   lv = lv, dwt_type = dwt_type, fixed_weights=fixed_weights)
    
    # Calculates for the error per pixel
    error = error_calculation(img_arr, reconst)
    print(error)
    
    return error

    
# run sim for non-v1 dct 
def run_sim_dct(method, observation, color, alpha, num_cell, img_arr, fixed_weights):
    ''' 
    Run a sim for non-v1 dct
    
    Parameters
    ----------

    method : String
        Method of reconstruction ('dwt' or 'dct')

    observation : String
        Method of observation (e.g. pixel, gaussian, v1)
    
    color : Boolean
        Desired mode to reconstruct image.
        True to reconstruct in RGB, False for grayscaled.

    alpha : float
        Penalty for fitting data onto LASSO function to 
        search for significant coefficents

    num_cell : List of int
        Number of blobs that will be used to be 
        determining which pixles to grab and use
    
    img_arr : numpy_array
        (n, m) shape image containing array of pixels

    Returns
    ----------
    error : float
        Computed normalized error value per each pixel
    '''
    dim = img_arr.shape
    if (len(dim) == 3) :
    	n, m, rgb = dim
    else :
    	n, m = dim
    # Deal with fraction num_cell amount
    if (num_cell < 1):
        num_cell = round(n * m * num_cell)
    num_cell = int(num_cell)
    img_arr = np.array([img_arr]).squeeze()
    reconst = large_img_experiment(img_arr, num_cell = num_cell, alpha = alpha,
                                   method = method, observation = observation,
                                   color = color, fixed_weights=fixed_weights)
    
    # Call function and calculate error
    error = error_calculation(img_arr, reconst)
    print(error)
    return error

# run sim for v1 dct
def run_sim_V1_dct(method, observation, color, alpha,
                   num_cell, cell_size, sparse_freq, img_arr, fixed_weights):
    ''' 
    Run a sim for V1 dct
    
    Parameters
    ----------
    method : String
        Method of reconstruction ('dwt' or 'dct')
    
    observation : String
        Method of observation (e.g. pixel, gaussian, v1)
    
    color : Boolean
        Desired mode to reconstruct image.
        True to reconstruct in RGB, False for grayscaled.

    alpha : float
        Penalty for fitting data onto LASSO function to 
        search for significant coefficents

    num_cell : int
        Number of blobs that will be used to be 
        determining which pixels to grab and use
    
    cell_size : int
        Determines field size of opened and closed blob of data. 
        Affect the data training

    sparse_freq : int
        Determines filed frequency on how frequently 
        opened and closed area would appear. Affect the data training
        
    img_arr : numpy_array
        (n, m) shape image containing array of pixels

    Returns
    ----------
    error : float
        Computed normalized error value per each pixel
    '''
    dim = img_arr.shape
    if (len(dim) == 3) :
    	n, m, rgb = dim
    else :
    	n, m = dim
    # Deal with fraction num_cell amount
    if (num_cell < 1):
        num_cell = round(n * m * num_cell)
    num_cell = int(num_cell)
    img_arr = np.array([img_arr]).squeeze()
    reconst = large_img_experiment(img_arr, num_cell = num_cell,
                                   cell_size=cell_size, sparse_freq=sparse_freq,
                                   alpha = alpha, method = method,
                                   observation = observation, color = color, fixed_weights=fixed_weights)
    error = error_calculation(img_arr, reconst)
    print(error)
    return error


def main():
    method, img, observation, color, dwt_type, level, alpha_list, \
        num_cell, cell_size, sparse_freq, fixed_weights = parse_sweep_args()
    run_sweep(method, img, observation, color, dwt_type, level, alpha_list,
              num_cell, cell_size, sparse_freq, fixed_weights)

if __name__ == '__main__':
    main()
