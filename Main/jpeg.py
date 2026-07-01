import struct
import os
import time
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image


from Encoder import encoding
from Decoder import decoding
from pre_enconde import downscale, get_greyscale_image

# --- 1. Metric Helper Functions ---
def rmse(img, rec):
    return np.sqrt(np.mean((img - rec)**2))

def psnr(img, rec):
    mse = np.mean((img - rec)**2)
    if mse == 0: return np.inf
    return 10 * np.log10(255**2 / mse)

def bpp(size_bytes, shape):
    n, m = shape[:2]
    return 8 * size_bytes / (n * m)

# --- 2. Fractal Size Calculator ---
def get_fractal_size(ifs, filename="fractal_code.bin"):
    """
    Flattens the nested ifs list and packs it into binary
    to calculate the real compressed file size.
    """
    with open(filename, 'wb') as f:
        # Loop through the nested list: ifs[row][col]
        for row in ifs:
            for block in row:
                if block is None: continue
                
                # block structure is: (k, l, t_id, s, o)
                k, l, t_id, s, o = block
                
                # Pack into binary:
                # 'H' = unsigned short (2 bytes) for coordinates k, l
                # 'B' = unsigned char (1 byte) for transformation ID
                # 'f' = float (4 bytes) for s and o
                # Total per block = 13 bytes
                try:
                    binary_data = struct.pack('HHBff', int(k), int(l), int(t_id), float(s), float(o))
                    f.write(binary_data)
                except Exception as e:
                    print(f"Error packing block {block}: {e}")

    return os.path.getsize(filename)

# --- 3. Main Execution ---
def generate_comparison_table():
    results = []
    
    # --- Load Reference Image ---
    # We load this just to get the original shape and calculate metrics
    # Adjust path if necessary
    #img_path = '../figures/test5.png'
    #if not os.path.exists(img_path):
    #    print(f"Error: Could not find image at {img_path}")
    #    return

    img = mpimg.imread('../figures/test5.png')
    img = get_greyscale_image(img)
    img = downscale(img,8)
    h, w = img.shape

    # ==========================================
    # PART A: FRACTAL (Run this first to get 'ifs')
    # ==========================================
    print("Running Fractal Compression...")
    
   
    
    d_size, r_size, stride = 8,4,8
    img_normalized = img.astype(np.float64) / 255.0
    t0 = time.time()
    ifs = encoding(img,d_size,r_size,stride)
    fractal_encode_time = time.time() - t0

    t0 = time.time()
    iterations = decoding(ifs, d_size, r_size, stride, n_iter=8)
    fractal_decode_time = time.time() - t0

    final_img_fractal = iterations[-1]
    final_img_fractal *= 255.0 
    plt.imshow(final_img_fractal,cmap='gray')
    # Calculate Real File Size
    fractal_bin_path = "temp_fractal.bin"
    fractal_size_bytes = get_fractal_size(ifs, fractal_bin_path)
    """ if final_img_fractal.shape != img.shape:
        print(f"⚠️ Warning: Resizing fractal image from {final_img_fractal.shape} to {img.shape}")
        
        # Calculate scale factor (e.g., 1024 / 128 = 8)
        scale_h = img.shape[0] // final_img_fractal.shape[0]
        scale_w = img.shape[1] // final_img_fractal.shape[1]
        
        # Use Kronecker product to scale up blocks (keeps the blocky fractal look)
        # equivalent to "nearest neighbor" upscaling
        final_img_fractal = np.kron(final_img_fractal, np.ones((scale_h, scale_w)))
    """
    results.append({
        "Method": "Fractal",
        "Block Size": f"{r_size}x{r_size}",
        "Size (KB)": fractal_size_bytes / 1024,
        "bpp": bpp(fractal_size_bytes, img.shape),
        "RMSE": rmse(img, final_img_fractal),
        "PSNR": psnr(img, final_img_fractal),
        "Encode (s)": fractal_encode_time,
        "Decode (s)": fractal_decode_time
    })

    # ==========================================
    # PART B: JPEG
    # ==========================================
    print("Running JPEG Compression...")
    jpeg_path = "../figures/test_c.jpg"
    Q = 50
    
    # Encode
    t0 = time.time()
    Image.fromarray(img.astype(np.uint8)).save(jpeg_path, quality=Q)
    jpeg_enc_time = time.time() - t0
    
    # Get Size
    jpeg_size_bytes = os.path.getsize(jpeg_path)
    
    # Decode
    t0 = time.time()
    jpeg_rec = np.array(Image.open(jpeg_path).convert('L'))
    jpeg_dec_time = time.time() - t0

    results.append({
        "Method": f"JPEG (Q={Q})",
        "Block Size": "--",
        "Size (KB)": jpeg_size_bytes / 1024,
        "bpp": bpp(jpeg_size_bytes, img.shape),
        "RMSE": rmse(img, jpeg_rec),
        "PSNR": psnr(img, jpeg_rec),
        "Encode (s)": jpeg_enc_time,
        "Decode (s)": jpeg_dec_time
    })

    # ==========================================
    # PART C: OUTPUT
    # ==========================================
    df = pd.DataFrame(results)
    
    # Reorder for clean look
    cols = ["Method", "Block Size", "Size (KB)", "bpp", "RMSE", "PSNR", "Encode (s)", "Decode (s)"]
    df = df[cols]

    print("\n--- Final Results ---")
    print(df)

    # Save to LaTeX
    latex_code = df.to_latex(
        index=False,
        float_format="%.4f",
        caption="Comparison: Fractal vs JPEG",
        label="tab:compression",
        column_format="lccccccc"
    )
    
    with open("compression_table.tex", "w") as f:
        f.write(latex_code)
        
    print("\nTable saved to 'compression_table.tex'")
    
    # Clean up temp file
    if os.path.exists(fractal_bin_path):
        os.remove(fractal_bin_path)

if __name__ == "__main__":
    generate_comparison_table()