import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
# Toy image
#I = np.zeros((64, 64), dtype=np.float64)
#I[16:48, 16:48] = 1.0

I = cv2.imread('test3.png',cv2.IMREAD_GRAYSCALE)
I = I.astype(np.float64)/255.0
n, m = I.shape

def extract_range_blocks(img, r=2):
    blocks = []
    for i in range(0, img.shape[0], r):
        for j in range(0, img.shape[1], r):
            blocks.append(((i, j), img[i:i+r, j:j+r]))
    return blocks


def extract_domain_blocks(img, d=4, step=2):
    blocks = []
    for i in range(0, img.shape[0] - d + 1, step):
        for j in range(0, img.shape[1] - d + 1, step):
            blocks.append(((i, j), img[i:i+d, j:j+d]))
    return blocks

def downscale(block):
    # 4x4 → 2x2 by averaging
    return block.reshape(2, 2, 2, 2).mean(axis=(2, 3))

def fit_affine(D, R):
    d = D.ravel()
    r = R.ravel()

    md = d.mean()
    mr = r.mean()

    vd = np.sum((d - md)**2)
    if vd < 1e-6:
        return 0.0, mr

    s = np.sum((d - md)*(r - mr)) / vd
    s = np.clip(s, -0.9, 0.9)
    o = mr - s*md

    return s, o

domain_blocks = extract_domain_blocks(I, d=4, step=2)

domain_data = []
for (i, j), D in domain_blocks:
    D_small = downscale(D)
    domain_data.append({
        "pos": (i, j),
        "block": D_small,
        "mean": D_small.mean(),
        "var":  D_small.var()
    })

range_blocks = extract_range_blocks(I, r=2)

VAR_TOL = 0.05
MEAN_TOL = 0.1

start = time.perf_counter()
ifs = []

for (i_r, j_r), R in range_blocks:
    mean_r = R.mean()
    var_r  = R.var()

    best_err = np.inf
    best_map = None

    for d in domain_data:
        # 🚫 Cheap rejection tests
        if abs(d["var"] - var_r) > VAR_TOL:
            continue
        if abs(d["mean"] - mean_r) > MEAN_TOL:
            continue

        s, o = fit_affine(d["block"], R)
        approx = s * d["block"] + o
        err = np.linalg.norm(R - approx)

        if err < best_err:
            best_err = err
            best_map = (d["pos"], s, o)

    ifs.append(best_map)

end = time.perf_counter()
print(f"Encoding time: {end - start:.3f} s")

def decode(ifs, n, m, n_iter=15):
    img = np.random.rand(n, m)

    for _ in range(n_iter):
        new = np.zeros_like(img)
        idx = 0

        for i in range(0, n, 2):
            for j in range(0, m, 2):
                (i_d, j_d), s, o = ifs[idx]
                D = img[i_d:i_d+4, j_d:j_d+4]
                new[i:i+2, j:j+2] = s * downscale(D) + o
                idx += 1

        img = np.clip(new, 0, 1)

    return img

J = decode(ifs, n, m)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(I, cmap="gray")
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(J, cmap="gray")
ax[1].set_title("Decoded")
ax[1].axis("off")

plt.show()
