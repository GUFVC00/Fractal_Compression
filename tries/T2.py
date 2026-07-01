import numpy as np
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def get_greyscale_image(img):
    return np.mean(img[:,:,:2], 2)

def downscale(img,factor):
    result = np.zeros((img.shape[0]//factor,img.shape[1]//factor))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i,j] = np.mean(img[i*factor:(i+1)*factor,j*factor:(j+1)*factor])
    
    return result

def  contrast_brightness(R,D):
    A = np.concatenate((np.ones((D.size, 1)),np.reshape(D,(D.size, 1))),axis=1)
    b = np.reshape(R,(R.size,))
    x,_,_,_ = np.linalg.lstsq(A,b)
    return x[1],x[0]

def match_size(img,d_size,r_size,stride):
    factor = d_size//r_size
    d_small = []
    for k in range(0, img.shape[0] - d_size + 1, stride):
        for l in range(0, img.shape[1] - d_size + 1, stride):
            D = downscale(
                img[k:k + d_size, l:l + d_size],
                factor
            )
            d_small.append((k, l, D))

    return d_small
"""
    for k in range((img.shape[0] - d_size)//(stride + 1)):
        for l in range((img.shape[1] - d_size)//(stride + 1)):
            d_small.append((k,l,downscale(img[k*stride:k*stride+d_size,l*stride:l*stride+d_size],factor)))
                   

    
    return d_small
"""
def encoding(img, d_size, r_size, stride):
    ifs = []
    d_small = match_size(img, d_size,r_size,stride)
    i_r = img.shape[0]//r_size
    j_r = img.shape[1]//r_size
    for i in range(i_r):
        ifs.append([])
        for j in range(j_r):
            ifs[i].append(None)
            min_ = float("inf")
            R = img[i*r_size:(i+1)*r_size,j*r_size:(j+1)*r_size]

            for k,l,D in d_small:
                s, o = contrast_brightness(R,D)
                s = np.clip(s,-1.0,1.0)
                D = s*D + o
                d = np.sum(np.square(R-D))
                if d < min_:
                    min_ = d
                    ifs[i][j] = (k,l,s,o)
    return ifs

def decoding(ifs,d_size,r_size,stride,n_iter = 8):
    factor = d_size//r_size
    h = len(ifs)*r_size
    w = len(ifs[0])*r_size
    J = [np.random.rand(
        h,w)]
    
    for i_ in range(n_iter):
        I = np.zeros((h,w))
        for i in range(len(ifs)):
            for j in range(len(ifs[0])):
                k,l,s,o = ifs[i][j]
                block = J[-1][k:k + d_size,l:l+d_size]
                if block.shape != (d_size,d_size):
                   continue
                D = downscale(block,factor)
                I[i*r_size:(i+1)*r_size,j*r_size:(j+1)*r_size] = np.clip(s*D + o,0.0,1.0)
        
        J.append(I)
  
    
    
    
    return  J




def plot_iterations(iterations, target=None):
    # Configure plot
    plt.figure()
    nb_row = math.ceil(np.sqrt(len(iterations)))
    nb_cols = nb_row
    # Plot
    for i, img in enumerate(iterations):
        print( img.min(), img.max())
        plt.subplot(nb_row, nb_cols, i+1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='none')
        if target is None:
            plt.title(str(i))
        else:
            # Display the RMSE
            plt.title(str(i) + ' (' + '{0:.2f}'.format(np.sqrt(np.mean(np.square(target - img)))) + ')')
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plt.tight_layout()

def test_greyscale():
    img = np.zeros((64,64))
    #img[1:59, 40:63] = 1.0
    
    img[16:48, 16:48] = 1.0
    img[24:40, 24:40] = 0.5

    #img = mpimg.imread('figures/test3.png')
    #img = get_greyscale_image(img)
    img = downscale(img, 4)

    

    plt.figure()
    plt.imshow(img, cmap='gray', vmin = 0,vmax = 1,interpolation='none')
    ifs = encoding(img, 8, 4, 8)
    iterations = decoding(ifs, 8, 4, 8)
    plot_iterations(iterations, img)
    plt.show()

test_greyscale()