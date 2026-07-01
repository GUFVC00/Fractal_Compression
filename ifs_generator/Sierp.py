import numpy as np
import matplotlib.pyplot as plt
import random

# IFS transformations
# Each tuple: (a, b, c, d, e, f) for x' = ax + by + e, y' = cx + dy + f
ifs = [
    (0.5, 0, 0, 0.5, 0, 0),
    (0.5, 0, 0, 0.5, 0.5, 0),
    (0.5, 0, 0, 0.5, 0.25, 0.5) # Example for a standard equilateral triangle
]

x, y = [0.0], [0.0]

# Generate 50,000 points
for _ in range(50000):
    t = random.choice(ifs)
    x_new = t[0]*x[-1] + t[1]*y[-1] + t[4]
    y_new = t[2]*x[-1] + t[3]*y[-1] + t[5]
    x.append(x_new)
    y.append(y_new)

plt.figure(figsize=(8, 8))
plt.scatter(x, y, s=0.1, c='teal', marker='.')
plt.axis('equal')
plt.axis('off')
plt.title("Sierpinski Triangle - Chaos Game Method")
plt.show()