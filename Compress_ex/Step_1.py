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
plt.savefig('Compress_ex\initial.png')
plt.show()