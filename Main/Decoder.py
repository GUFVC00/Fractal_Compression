import numpy as np
import time
from pre_enconde import downscale,generate_isometries





def decoding(ifs, d_size, r_size, stride, n_iter=8):

    factor = d_size // r_size
    h = len(ifs) * r_size
    w = len(ifs[0]) * r_size
    
    J = [np.random.rand(h, w)]
    
    for i_ in range(n_iter):
        I = np.zeros((h, w))
        for i in range(len(ifs)):
            for j in range(len(ifs[0])):
                k, l,t_id, s, o = ifs[i][j]
                
                # Extract block from previous iteration
                block = J[-1][k:k + d_size, l:l + d_size]
                
                # Handle edge cases where block might be smaller 
                if block.shape != (d_size, d_size):
                    continue
                
                D = downscale(block, factor)
                D = generate_isometries(D)[t_id]
                
                
                Im_rec = s * D + o
                I[i*r_size:(i+1)*r_size, j*r_size:(j+1)*r_size] = np.clip(Im_rec, 0.0, 1.0)
        
        J.append(I)

    return J
