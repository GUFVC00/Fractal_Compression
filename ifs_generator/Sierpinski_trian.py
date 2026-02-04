import numpy as np
import matplotlib.pyplot as plt

# Grid size
N = 100
t = np.zeros((N, N), dtype=int)
s = np.zeros((N, N), dtype=int)

# IFS transformations (a,b,c,d,e,f)
ifs = [
    (0.5, 0, 0, 0.5, 1, 1),
    (0.5, 0, 0, 0.5, 50, 1),
    (0.5, 0, 0, 0.5, 50, 50)
]

# Initial square border A(0)
t[0, :] = 1
t[-1, :] = 1
t[:, 0] = 1
t[:, -1] = 1

# Number of iterations
iterations = 20

for n in range(iterations):
    # Apply transformations to build s (A(n+1))
    for i in range(N):
        for j in range(N):
            if t[i, j] == 1:
                for (a, b, c, d, e, f) in ifs:
                    x = int(a*i + b*j + e)
                    y = int(c*i + d*j + f)
                    if 0 <= x < N and 0 <= y < N:
                        s[x, y] = 1
    
    # Copy s into t, reset s
    t[:,:] = s
    s.fill(0)

# Final plot of A(n+1)
plt.imshow(t.T, cmap="binary", origin="lower")
plt.title("Deterministic IFS Algorithm")
plt.show()
