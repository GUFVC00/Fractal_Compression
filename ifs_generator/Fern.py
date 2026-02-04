import numpy as np
import matplotlib.pyplot as plt


# Barnsley fern transformations
transforms = [
    (np.array([[0, 0], [0, 0.16]]), np.array([0, 0])),       # f1
    (np.array([[0.85, 0.04], [-0.04, 0.85]]), np.array([0, 1.6])), # f2
    (np.array([[0.20, -0.26], [0.23, 0.22]]), np.array([0, 1.6])), # f3
    (np.array([[-0.15, 0.28], [0.26, 0.24]]), np.array([0, 0.44])) # f4
]

# Probabilities
probs = [0.01, 0.85, 0.07, 0.07]

iterations = 100000

points = np.zeros((iterations,2))
x = np.array([0,0])

for i in range(iterations):
    index = np.random.choice(len(transforms), p = probs)
    A,b = transforms[index]
    x = A@x + b
    points[i] = x

plt.figure(figsize=(12,10))
plt.scatter(points[:,0],points[:,1], s =0.2, c="green")
plt.title("Barnsley Fern")
plt.axis("off")
plt.show()
