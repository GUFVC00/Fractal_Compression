import matplotlib.pyplot as plt

# IFS transformations for an equilateral Sierpinski triangle
ifs = [
    (0.5, 0, 0, 0.5, 0, 0),          # Bottom Left
    (0.5, 0, 0, 0.5, 0.5, 0),        # Bottom Right
    (0.5, 0, 0, 0.5, 0.25, 0.433)    # Top
]

# Start with a single point (the "seed"). It can be anywhere!
points = [(0.0, 0.0)]

# DANGER: Because this grows exponentially (3^n), keep this number around 8 or 9.
# 9 iterations will perfectly generate 3^9 = 19,683 points.
iterations = 9

for n in range(iterations):
    new_points = []
    
    # Apply all 3 rules to EVERY point from the previous generation
    for (x, y) in points:
        for (a, b, c, d, e, f) in ifs:
            new_x = a * x + b * y + e
            new_y = c * x + d * y + f
            new_points.append((new_x, new_y))
            
    # Overwrite the old generation with the new one
    points = new_points

# Separate the tuples into X and Y lists for the scatter plot
x_vals = [p[0] for p in points]
y_vals = [p[1] for p in points]

# Plot the results exactly like we did for the random method
plt.figure(figsize=(8, 8))
plt.scatter(x_vals, y_vals, s=0.5, c='black', marker='.')
plt.axis('equal')
plt.axis('off')
plt.title(f" Deterministic algorithm ({len(points)} points)")
plt.savefig('Sierp_tria.jpeg')
plt.show()