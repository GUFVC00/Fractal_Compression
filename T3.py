import numpy as np
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


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
                
                # Handle edge cases where block might be smaller (though shouldn't happen with correct logic)
                if block.shape != (d_size, d_size):
                    continue
                
                D = downscale(block, factor)
                D = generate_isometries(D)[t_id]
                
                
                Im_rec = s * D + o
                I[i*r_size:(i+1)*r_size, j*r_size:(j+1)*r_size] = np.clip(Im_rec, 0.0, 1.0)
        
        J.append(I)
    
    return J

def plot_iterations(iterations, target=None):
    plt.figure(figsize=(10, 10))
    nb_row = math.ceil(np.sqrt(len(iterations)))
    nb_cols = nb_row
    
    for i, img in enumerate(iterations):
        plt.subplot(nb_row, nb_cols, i+1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1.0, interpolation='none')
        
        if target is None:
            plt.title(str(i))
        else:
            rmse = np.sqrt(np.mean(np.square(target - img)))
            plt.title(f'{i} (RMSE: {rmse:.4f})')
            
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plt.tight_layout()

def test_greyscale():
   
    #img = np.zeros((128, 128))
    #img[16:48, 1:48] = 1.0
    #img[24:40, 24:40] = 0.5 

   
    img = mpimg.imread('figures/test2.png')
    if len(img.shape) > 2:
         img = get_greyscale_image(img)
         img = downscale(img,8)


    plt.figure()
    plt.title("Original Target")
    plt.imshow(img, cmap='gray', vmin=0, vmax=1.0)
    
    # Adjusted parameters for the 64x64 image
    # d_size (Domain) = 8, r_size (Range) = 4 -> Compression factor
    ifs = encoding(img, d_size=8, r_size=4, stride=8)
    iterations = decoding(ifs, d_size=8, r_size=4, stride=8, n_iter=8)
    
    plot_iterations(iterations, img)
    plt.show()

if __name__ == "__main__":
    test_greyscale()