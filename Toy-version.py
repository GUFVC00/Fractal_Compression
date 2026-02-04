import numpy as np
import matplotlib.pyplot as plt 

I = np.array([
    [0,0,0,0,1,1,1,1],
    [0,0,0,0,1,1,1,1],
    [0,0,0,0,1,1,1,1],
    [0,0,0,0,1,1,1,1],
    [1,1,1,1,0,0,0,0],
    [1,1,1,1,0,0,0,0],
    [1,1,1,1,0,0,0,0],
    [1,1,1,1,0,0,0,0],
], dtype=float)

plt.imshow(I,cmap = 'gray')
plt.title('Original image')
plt.colorbar()
plt.show()

N = I.shape[0]  #The height of the image
M = I.shape[1]  #The width of the image
block_size = 2

def extract_block(img, block_size):
    blocks = []
    N = img.shape[0]  #The height of the image
    M = img.shape[1]  #The width of the image
    for i in range(0,N, block_size):
        for j in range(0,M, block_size):
            blocks.append(img[i:i+block_size, j:j+block_size])
    return blocks

range_blocks = extract_block(I,2)
domain_blocks = extract_block(I,4)

def downscale(block):
    return block.reshape(2,2,2,2).mean(axis=(2,3))

def fit_affine(D, R):
    d = D.flatten()
    r = R.flatten()
    
    A = np.vstack([d, np.ones(len(d))]).T
    s, o = np.linalg.lstsq(A, r, rcond=None)[0]
    return s, o

ifs = []

for R in range_blocks:
    best_error = np.inf
    best_map = None

    for k, D in enumerate(domain_blocks):
        D_small = downscale(D)
        s, o = fit_affine(D_small, R)
        approx = s * D_small + o
        error = np.linalg.norm(R - approx)

        if error < best_error:
            best_error = error
            best_map = (k, s, o)

    ifs.append(best_map)
J = np.random.rand(8,8)

def decode(ifs, domain_blocks, n_iter=10):
    img = np.random.rand(8,8)

    for _ in range(n_iter):
        new_img = np.zeros_like(img)
        idx = 0

        for i in range(0, 8, 2):
            for j in range(0, 8, 2):
                k, s, o = ifs[idx]
                D = extract_block(img, 4)[k]
                D_small = downscale(D)
                new_img[i:i+2, j:j+2] = s * D_small + o
                idx += 1

        img = new_img

    return img

J = decode(ifs, domain_blocks, n_iter=15)

plt.imshow(J, cmap="gray")
plt.title("Decoded Image")
plt.colorbar()
plt.show()



# 1. Define grid dimensions 
# (You need the original image dimensions N, M and block_size)
grid_h = N // block_size
grid_w = M // block_size

# 2. Unpack the tuples
# ifs is a list of tuples: (index, contrast, offset)
indices = [item[0] for item in ifs]
contrasts = [item[1] for item in ifs]
offsets = [item[2] for item in ifs]

# 3. Reshape into 2D grids
map_k = np.array(indices).reshape(grid_h, grid_w)
map_s = np.array(contrasts).reshape(grid_h, grid_w)
map_o = np.array(offsets).reshape(grid_h, grid_w)

# 4. Plot
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Plot Source Index Map
im1 = ax[0].imshow(map_k, cmap='jet')
ax[0].set_title("Source Block Index (k)")
plt.colorbar(im1, ax=ax[0])

# Plot Contrast Map
im2 = ax[1].imshow(map_s, cmap='coolwarm', vmin=-1, vmax=1)
ax[1].set_title("Contrast Scaling (s)")
plt.colorbar(im2, ax=ax[1])

# Plot Offset Map
im3 = ax[2].imshow(map_o, cmap='gray')
ax[2].set_title("Brightness Offset (o)")
plt.colorbar(im3, ax=ax[2])

plt.show()