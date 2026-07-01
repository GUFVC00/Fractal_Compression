import numpy as np
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import cv2
from numpy.lib.stride_tricks import sliding_window_view

def get_greyscale_image(img):
    # Using standard luminance weights yields better perceptual greyscale than pure mean
    if img.shape[-1] == 4: # Handle RGBA
        img = img[:, :, :3]
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

def downscale(img, factor):
    h, w = img.shape
    return img.reshape(h//factor, factor, w//factor, factor).mean(axis=(1, 3))

def decoding_zoom(ifs, original_width, original_height, d_size, r_size, scale_factor, iterations=8):
    # 1. Calculate the new, higher-resolution dimensions
    new_width = original_width * scale_factor
    new_height = original_height * scale_factor
    
    # 2. Scale up the block sizes
    new_r_size = r_size * scale_factor
    new_d_size = d_size * scale_factor
    
    # 3. Create the starting canvas (can be all zeros, mid-gray, or random noise)
    # The canvas MUST be the new, higher resolution.
    decoded_img = np.ones((new_height, new_width)) * 128 
    
    # The grid dimensions of the IFS remain exactly the same!
    i_r = len(ifs)
    j_r = len(ifs[0])
    
    for iteration in range(iterations):
        print(f"Decoding Zoom Iteration: {iteration + 1}/{iterations}")
        
        # Create a buffer so we don't overwrite data we are currently reading
        new_img = np.zeros_like(decoded_img)
        
        for i in range(i_r):
            for j in range(j_r):
                # Unpack your stored IFS parameters
                k, l, t_id, s, o = ifs[i][j]
                
                # 4. SCALE THE DOMAIN COORDINATES
                # Assuming k and l are the top-left pixel coordinates of the domain block 
                # in the ORIGINAL image, you must scale them up to find the block in the NEW image.
                new_k = k * scale_factor
                new_l = l * scale_factor
                
                # Extract the larger Domain block from the current image
                # ... [previous code] ...
                # Extract the larger Domain block from the current image
                D = decoded_img[new_k : new_k + new_d_size, new_l : new_l + new_d_size]
                
                # 5. Shrink the Domain block to the new Range block size
                D_shrunk = cv2.resize(D, (new_r_size, new_r_size), interpolation=cv2.INTER_AREA)
                
                # Apply the spatial transformation using your exact t_id logic!
                D_transformed = np.rot90(D_shrunk, t_id // 2)
                if t_id % 2 == 1:
                    D_transformed = np.fliplr(D_transformed)
                
                # Apply contrast and brightness (make sure to use D_transformed!)
                R_approx = s * D_transformed + o
                
                # 6. Place the new block onto the scaled canvas
                # Clip the values just like you do in your standard decoder
                new_img[i * new_r_size : (i+1) * new_r_size, j * new_r_size : (j+1) * new_r_size] = np.clip(R_approx, 0.0, 1.0)
        # Update the image for the next iteration
        decoded_img = new_img
        
    return decoded_img

def encoding_optimized(img, d_size, r_size, stride):
    encd_time = time.perf_counter()
    
    factor = d_size // r_size
    n = r_size * r_size
    
    # 1. Extract ALL Domain blocks simultaneously using a sliding window
    D_windows = sliding_window_view(img, (d_size, d_size))[::stride, ::stride]
    h_w, w_w = D_windows.shape[:2]
    
    # Flatten spatial window dimensions: shape becomes (N_D, d_size, d_size)
    D_blocks = D_windows.reshape(-1, d_size, d_size)
    
    # 2. Batch downscale all domain blocks at once
    D_down = D_blocks.reshape(-1, r_size, factor, r_size, factor).mean(axis=(2, 4))
    
    # 3. Generate all 8 isometries for all blocks concurrently
    D_isometries = []
    for k in range(4):
        rot = np.rot90(D_down, k, axes=(1, 2))
        D_isometries.append(rot)
        D_isometries.append(np.flip(rot, axis=2)) # fliplr equivalent for 3D array
        
    # Stack and flatten for matrix math: shape becomes (N_D * 8, r_size * r_size)
    D_all = np.stack(D_isometries, axis=1)
    D_flat = D_all.reshape(-1, n)
    
    # Precompute statistics for all domain blocks
    sum_D = D_flat.sum(axis=1)
    sum_D2 = (D_flat ** 2).sum(axis=1)
    
    # Denominator for 's'. Add a tiny epsilon to prevent division by zero errors
    denom = n * sum_D2 - sum_D ** 2
    safe_denom = np.where(denom == 0, 1e-10, denom)
    
    i_r = img.shape[0] // r_size
    j_r = img.shape[1] // r_size
    
    ifs = [[None] * j_r for _ in range(i_r)]
    total_domains = D_flat.shape[0]
    print(f"Total domain isometries to search per range: {total_domains}")
    
    # 4. Search loop: Iterates over Ranges, but Vectorized over Domains
    for i in range(i_r):
        for j in range(j_r):
            R = img[i*r_size:(i+1)*r_size, j*r_size:(j+1)*r_size]
            R_flat = R.flatten()
            sum_R = R_flat.sum()
            
            # Fast matrix dot product calculates sum_RD for ALL domains at once
            sum_RD = np.dot(D_flat, R_flat)
            
            # Calculate s and o arrays (one value per domain block)
            s = (n * sum_RD - sum_R * sum_D) / safe_denom
            s = np.where(denom == 0, 0.0, s)
            s = np.clip(s, -1.0, 1.0)
            
            o = (sum_R - s * sum_D) / n
            
            # Reconstruct and calculate mean squared error across all domains simultaneously
            # Broadcasting sizes: s[:, None] is (total_domains, 1), D_flat is (total_domains, n)
            Aprox = s[:, None] * D_flat + o[:, None]
            errors = np.sum((Aprox - R_flat) ** 2, axis=1)
            
            # Extract the lowest error
            best_idx = np.argmin(errors)
            
            # Map the 1D best index back to k, l spatial coords and transformation ID
            orig_d_idx = best_idx // 8
            t_id = best_idx % 8
            k = (orig_d_idx // w_w) * stride
            l = (orig_d_idx % w_w) * stride
            
            ifs[i][j] = (k, l, t_id, s[best_idx], o[best_idx])
            
        print(f"Row {i+1}/{i_r} encoded.", end='\r')
        
    print(f"\nEncoding finished in {time.perf_counter() - encd_time:.2f} seconds.")
    return ifs

def decoding(ifs, d_size, r_size, stride, n_iter=8):
    dec_time = time.perf_counter()
    factor = d_size // r_size
    h = len(ifs) * r_size
    w = len(ifs[0]) * r_size
    
    J = [np.random.rand(h, w)]
    
    for _ in range(n_iter):
        I = np.zeros((h, w))
        for i in range(len(ifs)):
            for j in range(len(ifs[0])):
                k, l, t_id, s, o = ifs[i][j]
                
                # Extract domain block from previous iteration
                block = J[-1][k:k + d_size, l:l + d_size]
                if block.shape != (d_size, d_size):
                    continue
                
                # Downscale
                D = downscale(block, factor)
                
                # Apply the specific recorded isometry
                # t_id mapped back: t_id // 2 = rotations, t_id % 2 = flip
                D = np.rot90(D, t_id // 2)
                if t_id % 2 == 1:
                    D = np.fliplr(D)
                
                Im_rec = s * D + o
                I[i*r_size:(i+1)*r_size, j*r_size:(j+1)*r_size] = np.clip(Im_rec, 0.0, 1.0)
                
        J.append(I)
        
    print(f"Decoding finished in {time.perf_counter() - dec_time:.2f} seconds.")
    return J

def plot_iterations(iterations, target=None):
    plt.figure(figsize=(12, 12))
    nb_row = math.ceil(np.sqrt(len(iterations)))
    
    for i, img in enumerate(iterations):
        plt.subplot(nb_row, nb_row, i+1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1.0, interpolation='none')
        
        if target is None:
            plt.title(f"Iter {i}")
        else:
            h, w = img.shape

# Crop the target to match the reconstructed image's dimensions
            target_cropped = target[:h, :w]

# Calculate RMSE using the cropped target
            rmse = np.sqrt(np.mean(np.square(target_cropped - img)))
           # rmse = np.sqrt(np.mean(np.square(target - img)))
            plt.title(f'Iter {i} (RMSE: {rmse:.4f})')
            
        plt.axis('off')
    plt.tight_layout()

def test_greyscale():

    try:
        img = mpimg.imread('../figures/test2.png')
        if len(img.shape) >= 3:
            img = get_greyscale_image(img)


        

    except FileNotFoundError:
        # Fallback to a synthetic gradient image for testing if the file isn't found
        print("Image not found. Generating a synthetic test image.")
        img = np.linspace(0, 1, 64).reshape(-1, 1) * np.linspace(0, 1, 64).reshape(1, -1)
        img[16:48, 16:48] = 0.5
        
    plt.figure()
    plt.title("Original Target")
    plt.imshow(img, cmap='gray', vmin=0, vmax=1.0)
    plt.axis('off')
    
   # ... [your existing encoding and standard decoding code] ...
    ifs = encoding_optimized(img, d_size=4, r_size=2, stride=2) 
    iterations = decoding(ifs, d_size=4, r_size=2, stride=2, n_iter=8)
    
    plot_iterations(iterations, img)
    #plt.savefig('figures/fern_comp')
    plt.show()

    # --- ADD THIS TO TEST THE ZOOM ---
    print("\n--- Starting Fractal Zoom (2x Resolution) ---")
    # We pass a scale_factor of 2 to double the resolution
    zoomed_img = decoding_zoom(
        ifs, 
        original_width=img.shape[1], 
        original_height=img.shape[0], 
        d_size=8, 
        r_size=4, 
        scale_factor=2, 
        iterations=8
    )
    
    plt.figure(figsize=(8, 8))
    plt.title("2x Resolution Zoomed Fractal Image")
    plt.imshow(zoomed_img, cmap='gray', vmin=0, vmax=1.0)
    plt.axis('off')
    #plt.savefig('fern_zoom')
    plt.show()

if __name__ == "__main__":
    test_greyscale()