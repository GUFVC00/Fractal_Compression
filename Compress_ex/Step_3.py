import numpy as np
from Step_2 import extract,downscale, ifs,domain_blocks
import matplotlib.pyplot as plt


def decode(ifs,domain_block, n_iter):
    img = np.random.rand(8,8)

    for l in range(n_iter):
        new_img = np.zeros_like(img)
        idx = 0

        for i in range(0,8,2):
            for j in range(0,8,2):
                k,s,o = ifs[idx]
                D = extract(img,4)[k]
                D_small = downscale(D)
                new_img[i:i+2,j:j+2] = s*D_small + o
                idx += 1

        img = new_img
    return img

J = decode(ifs, domain_blocks, n_iter=15)

plt.imshow(J, cmap="gray")
plt.title("Decoded Image")
plt.colorbar()
plt.show()