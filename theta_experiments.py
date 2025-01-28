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
cell_size = 7
sparse_freq = 2

## For wavelet variable
lv= 2
dwt_type= 'db2'

#Load Images:
# Represent image as numpy array to make it easier to process
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))

small_img_arr = process_image(small_img, mode)
ax1.imshow(small_img_arr)
ax1.set_title("{img}".format(img = small_img.split('.')[0]))
ax1.axis('off')
print(small_img_arr.shape)
small_img_arr_gray = process_image(small_img, False) #change from 'gray' to False
ax2.imshow(small_img_arr_gray, 'gray')
print(small_img_arr_gray.shape)
ax2.set_title("{img} grayscaled".format(img = small_img.split('.')[0]))
ax2.axis('off')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
big_img_arr = process_image(big_img, mode)
ax1.imshow(big_img_arr)
ax1.set_title("{img}".format(img = big_img.split('.')[0]))
ax1.axis('off')
print(big_img_arr.shape)
big_img_arr_gray = process_image(big_img, False) #change from 'gray' to False
ax2.imshow(big_img_arr_gray, 'gray')
ax2.set_title("{img} grayscaled".format(img = big_img.split('.')[0]))
ax2.axis('off')
print(big_img_arr_gray.shape)
plt.show()

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
plt.show()
