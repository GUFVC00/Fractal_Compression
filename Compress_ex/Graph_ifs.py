import matplotlib.pyplot as plt
import numpy as np
from Step_2 import ifs

# 1. Prepare the data (Get the first 15 blocks to keep it readable)
# Assuming 'ifs' is your list of tuples: [(k, s, o), (k, s, o), ...]
data_to_show = ifs[:15]

# Convert the data into a format suitable for the table (Lists of strings)
# We add the "Block Index" (i) manually
cell_text = []
for i, (k, s, o) in enumerate(data_to_show):
    # Format numbers to 2 decimal places for cleaner look
    cell_text.append([str(i), str(k), f"{s:.2f}", f"{o:.2f}"])

# 2. Define Column Headers
columns = ("Block Index", "Source ID (k)", "Contrast (s)", "Offset (o)")

# 3. Create the Plot
fig, ax = plt.subplots(figsize=(8, 6))

# Hide actual graph axes (we just want the table)
ax.axis('tight')
ax.axis('off')

# 4. Create the Table
the_table = ax.table(cellText=cell_text,
                     colLabels=columns,
                     loc='center',
                     cellLoc='center')

# 5. Styling (Optional but recommended)
the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
the_table.scale(1, 1.5)  # Stretch the rows slightly for readability

# Color the header row
for (row, col), cell in the_table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#40466e') # Dark blue header

plt.title("First 15 IFS Transforms", y=1.02)
plt.savefig("Compress_ex\IFS.png")
plt.show()