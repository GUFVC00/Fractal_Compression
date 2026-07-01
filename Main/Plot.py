import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.image as mpimg


from Encoder import encoding
from Decoder import decoding
from pre_enconde import get_greyscale_image,downscale



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

   
    img = mpimg.imread('../figures/test5.png')
    if len(img.shape) > 2:
         img = get_greyscale_image(img)
         img = downscale(img,8)


    plt.figure()
    plt.title("Original Target")
    plt.imshow(img, cmap='gray', vmin=0, vmax=1.0)
    
    # Adjusted parameters for the 64x64 image
    # d_size (Domain) = 8, r_size (Range) = 4 -> Compression factor
    d_size = 8
    r_size = 4
    ifs = encoding(img, d_size=8, r_size=4, stride=8)
    iterations = decoding(ifs, d_size=8, r_size=4, stride=8, n_iter=8)
    
    plot_iterations(iterations, img)
    plt.show()
    final_img = iterations[-1]
    return final_img,ifs,d_size,r_size

test_greyscale()