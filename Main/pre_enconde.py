import numpy as np
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import pandas as pd
import os



def get_greyscale_image(img):                        #Guarantee that the figure is on grayscale
    return np.mean(img[:,:,:2], 2)

def downscale(img, factor):
    # Change the quantity of pixels to accelerate the enconding process
    h, w = img.shape
    return img.reshape(h//factor, factor, w//factor, factor).mean(axis=(1, 3))


def get_block_stats(block):
    """Calculates pre-computed stats to speed up matching"""
    flat = block.flatten()
    sum_d = np.sum(flat)
    sum_dd = np.sum(flat ** 2)
    return sum_d, sum_dd, flat

def generate_isometries(block):
    transforms = []
    for k in range(4):
        rot = np.rot90(block,k)
        transforms.append(np.fliplr(rot))
    return transforms

def match_size(img, d_size, r_size, stride):
    factor = d_size // r_size
    d_small = []
    
    # Pre-allocate to save memory/time
    # We step through the image and immediately downscale
    for k in range(0, img.shape[0] - d_size + 1, stride):
        for l in range(0, img.shape[1] - d_size + 1, stride):
            # Extract
            block = img[k:k + d_size, l:l + d_size]
            # Downscale
            D0 = downscale(block, factor)

            for t_id, D in enumerate(generate_isometries(D0)):
                sum_d, sum_dd, D_flat = get_block_stats(D)
                d_small.append((k, l,t_id, D, sum_d, sum_dd, D_flat))
            
    return d_small
