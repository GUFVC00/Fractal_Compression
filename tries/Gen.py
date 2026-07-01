import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

#I = cv2.imread('test3.png',cv2.IMREAD_GRAYSCALE)
#I = I.astype(np.float64)/255.0
I = np.zeros((64,64))
I[1:2, 39:41] = 1.0

"""
plt. imshow(I,cmap = 'gray')
plt.title('Original image')
plt.show()
"""

n = I.shape[0]  #The height of the image
m = I.shape[1]  #The width of the image
block_size = 2


def extract_rblock(img,r):
    n,m = img.shape
    blocks = []
    for i in range(0,n-r+1, r):
        for j in range(0,m-r+1, r):
            blocks.append(((i,j),img[i:i+r, j:j+r]))
    return blocks

def extract_dblock(img,d):
    n,m = img.shape
    blocks = []

    for i in range(0,n - d + 1,2):
        for j in range(0,m - d + 1,2):
            blocks.append(((i,j), img[i:i+d,j:j+d]))
    return blocks

range_blocks = extract_rblock(I,2)
domain_blocks = extract_dblock(I,4)


def downscale(block):
    n,m = block.shape
    assert n % 2 == 0 and m % 2 == 0
   
    small = np.zeros((n//2,m//2))
    for i in range(0,n//2):
        for j in range (0,m//2):
            small[i,j] =  block[2*i:2*i+2, 2*j:2*j+2].mean()
    return small

def fit_affine(D, R):
    
    d = D.flatten()
    r = R.flatten()

    mean_d = d.mean()
    mean_r = r.mean()

    var_d = np.sum((d - mean_d)**2) 
    
    s = np.sum((d - mean_d)*(r - mean_r))/var_d
    s = np.clip(s,-0.99,0.99)
    o = mean_r - s*mean_d
    
    return s, o

print(len(range_blocks), len(domain_blocks))


start_time = time.perf_counter()

ifs = []


#try
domain_data = []
for ((i_d, j_d), D) in domain_blocks:
    D_small = downscale(D)
    var = np.var(D_small)
    domain_data.append((i_d, j_d, D_small, var))
#---

for (l,R) in range_blocks:
    best_error = np.inf
    best_map = None
    
    for (i_d, j_d, D_small, var_D) in domain_data:
        if var_D < 1e-6:
           continue
        s, o = fit_affine(D_small, R)
        approx = s * D_small + o
        error = np.linalg.norm(R - approx)



        if error < best_error:
            best_error = error
            best_map = (i_d,j_d, s, o)

    ifs.append(best_map)

end_time = time.perf_counter()

J = np.random.rand(n,m)


start_time2 = time.perf_counter()
def decode(ifs, domain_blocks,n,m,n_iter = 15):
    img = np.random.rand(n,m)

    for _ in range(n_iter):
        new_img = np.zeros_like(img)
        idx = 0
        
        domain_blocks = extract_dblock(img,4)

        for i in range(0, n, 2):
            for j in range(0, m, 2):
                i_d,j_d, s, o = ifs[idx]
                D = img[i_d:i_d+4,j_d:j_d+4]
                D_small = downscale(D)
                new_img[i:i+2, j:j+2] = s * D_small + o
                idx += 1

        img = np.clip(new_img, 0, 1)

    return img

J = decode(ifs, domain_blocks,n,m,15)
end_time2 = time.perf_counter()

#plt.imshow(J, cmap="gray")
#plt.title("Decoded Image")
#plt.colorbar()
#plt.show()


elapsed_time = end_time - start_time
elapsed_time2 = end_time2 - start_time2

print(f"Enconding time: {elapsed_time:.4f} seconds")
print(f"Decoding time: {elapsed_time2:.4f} seconds")




fig,axs = plt.subplots(1,2, figsize =(10,5))


axs[0].imshow(I,cmap = 'gray')
axs[0].set_title('Original')
axs[0].axis('off')

axs[1].imshow(J,cmap ='gray')
axs[1].set_title("Decoded")
axs[1].axis('off')

plt.suptitle(f"Enconding time: {elapsed_time:.4f} seconds and decoding time: {elapsed_time2:.4f} seconds", fontsize =14)


plt.tight_layout()
plt.show()