from Step_1 import I
import numpy as np
import matplotlib.pyplot as plt

def extract(img,block_size):
    blocks = []
    N = img.shape[0]   #height
    M = img.shape[1]   #width
    for i in range(0,N,block_size):
        for j in range(0,M,block_size):
            blocks.append(img[i:i+block_size,j:j+block_size])          #goes block by block
    return blocks

range_blocks = extract(I,2)
domain_blocks = extract(I,4)

def downscale(block):
    return block.reshape(2,2,2,2).mean(axis=(2,3))           #Contractivity function geometrically

def fit_affine(D,R):
    d = D.flatten()
    r = R.flatten()
    A = np.vstack([d,np.ones(len(d))]).T
    s,o = np.linalg.lstsq(A,r,rcond = None)[0]       #s -> contrast ,   o -> brightness
    return s , o

ifs = []

for R in range_blocks:
    best_err = np.inf
    best_map = None

    for k in range(len(domain_blocks)):
        D = domain_blocks[k]              
        D_small = downscale(D)                  #Match the geometric size
        s, o = fit_affine(D_small,R)            #Best constrast and brightness adjustment
        approx = s*D_small + o
        err = np.linalg.norm(approx - R)         

        if err < best_err:
           best_err = err
           best_map = (k,s,o)

    ifs.append(best_map)

