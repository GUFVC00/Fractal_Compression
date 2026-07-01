import numpy as np
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
import sys
from pre_enconde import match_size



def encoding(img, d_size, r_size, stride):
    ifs = []
    # 1. Pre-process all domain blocks ONCE
    d_small = match_size(img, d_size, r_size, stride)
    
    i_r = img.shape[0] // r_size
    j_r = img.shape[1] // r_size
    
    for i in range(i_r):
        ifs.append([])
        for j in range(j_r):
            ifs[i].append(None)
            min_error = float("inf")
            print("{}/{} ; {}/{}".format(i, i_r, j, j_r))
            # Extract Range Block
            R = img[i*r_size:(i+1)*r_size, j*r_size:(j+1)*r_size]
            R_flat = R.flatten()
            sum_r = np.sum(R_flat)
            n = R_flat.size
            
            # Search through pre-computed Domain blocks
            for k, l,t_id, D, sum_d, sum_dd, D_flat in d_small:
                
                # Optimized contrast/brightness calculation inline
                # We reuse sum_d and sum_dd from the list!
                sum_rd = np.sum(R_flat * D_flat)
                
                denominator = n * sum_dd - sum_d ** 2
                if denominator == 0:
                    s = 0.0
                else:
                    s = (n * sum_rd - sum_r * sum_d) / denominator
                
                s = np.clip(s, -1.0, 1.0)  
                o = (sum_r - s * sum_d) / n
                Aprox = s * D + o
                error = np.sum((R - Aprox) ** 2)
                
                if error < min_error:
                    min_error = error
                    ifs[i][j] = (k, l,t_id, s, o)
             
    return ifs
finish_encd = time.perf_counter()  
  


