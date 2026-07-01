import struct
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from Opt1 import downscale,encoding_optimized,decoding


# --- 1. Metric Helper Functions ---
def rmse(img, rec):
    # CRITICAL FIX: Convert to float64 before subtracting to prevent 8-bit wrap-around!
    return np.sqrt(np.mean((img - rec)**2))

def psnr(img, rec):
    mse = np.mean((img.astype(np.float64) - rec.astype(np.float64))**2)
    if mse == 0: return np.inf
    return 10 * np.log10(255**2 / mse)

def bpp(size_bytes, shape):
    n, m = shape[:2]
    return 8 * size_bytes / (n * m)

# --- 2. Fractal Size Calculator ---
def get_fractal_size(ifs, filename="fractal_code.bin"):
    with open(filename, 'wb') as f:
        for row in ifs:
            for block in row:
                if block is None: continue
                k, l, t_id, s, o = block
                try:
                    binary_data = struct.pack('HHBff', int(k), int(l), int(t_id), float(s), float(o))
                    f.write(binary_data)
                except Exception as e:
                    print(f"Error packing block {block}: {e}")
    return os.path.getsize(filename)

# --- 3. Main Execution ---
def generate_comparison_table():
    results = []
    
    # --- Load Reference Image Safely ---
    # Using PIL ensures the image is loaded strictly as 0-255 integers (unlike mpimg)
    img_path = '../figures/test2.png'
    raw_img = np.array(Image.open(img_path).convert('L'))
    
    # Apply your preprocessing
    img = downscale(raw_img, 1)

    # CRITICAL FIX: crop to a multiple of r_size so the fractal grid
    # (i_r * r_size, j_r * r_size) exactly matches img.shape. Without this,
    # any leftover rows/cols get silently dropped by the fractal encoder,
    # causing a shape mismatch in rmse/psnr later on.
    r_size_crop = 4  # must match r_size used below
    h, w = img.shape
    h = h - (h % r_size_crop)
    w = w - (w % r_size_crop)
    img = img[:h, :w]
    h, w = img.shape

    # ==========================================
    # PART A: FRACTAL 
    # ==========================================
    print("Running Fractal Compression...")
    
    d_size, r_size, stride = 8, 4, 8
    
    # Normalize purely for the fractal encoder math
    img_normalized = img.astype(np.float64) / 255.0
    
    t0 = time.time()
    # CRITICAL FIX: Pass the normalized image, not the raw one!
    ifs = encoding_optimized(img_normalized, d_size, r_size, stride)
    fractal_encode_time = time.time() - t0

    t0 = time.time()
    iterations = decoding(ifs, d_size, r_size, stride, n_iter=8)
    fractal_decode_time = time.time() - t0

    # Extract final image, scale it back to 0-255, and format it properly
    final_img_fractal = iterations[-1] * 255.0 
    final_img_fractal = np.clip(final_img_fractal, 0, 255).astype(np.uint8)
    
    # Calculate Real File Size
    fractal_bin_path = "temp_fractal.bin"
    fractal_size_bytes = get_fractal_size(ifs, fractal_bin_path)

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
    # CRITICAL FIX: never write to img_path! Doing so overwrote the original
    # source image with a downscaled/compressed copy on every run, so each
    # subsequent run started from an already-degraded image.
    jpeg_path = "../figures/test2_jpeg_out.jpg"
    Q = 50
    
    # Encode
    t0 = time.time()
    # img is now guaranteed to be valid 0-255 data
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
    
    cols = ["Method", "Block Size", "Size (KB)", "bpp", "RMSE", "PSNR", "Encode (s)", "Decode (s)"]
    df = df[cols]

    print("\n--- Final Results ---")
    print(df)

    latex_code = df.to_latex(
        index=False,
        float_format="%.4f",
        position= "h",
        caption="Comparison: Fractal vs JPEG",
        label="tab:compression",
        column_format="lccccccc"
    )
    
    with open("compression_table_6op.tex", "w") as f:
        f.write(latex_code)
        
    print("\nTable saved to 'compression_table.tex'")
    
    if os.path.exists(fractal_bin_path):
        os.remove(fractal_bin_path)

    # Optional: Plot the visual comparison
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1); plt.title("Original"); plt.imshow(img, cmap='gray'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.title("Fractal"); plt.imshow(final_img_fractal, cmap='gray'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.title(f"JPEG (Q={Q})"); plt.imshow(jpeg_rec, cmap='gray'); plt.axis('off')
    plt.tight_layout()
    #plt.savefig("figures/3_img_sidebyside6.jpg")
    plt.show()
    

if __name__ == "__main__":
    generate_comparison_table()