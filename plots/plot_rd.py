import numpy as np
import matplotlib.pyplot as plt

# Define the function
def rd_function(dist, THRESHOLD=0.1):
    if dist <= THRESHOLD:
        rd = 100 * (1 - (dist/THRESHOLD))**.5
    else:
        rd = (1 - (dist/THRESHOLD))
    return rd

# Generate the data
THRESHOLD = 0.1
dist_range = np.linspace(0, 1, 1000) # 1000 equally spaced points between 0 and 1
rd_values = [rd_function(dist, THRESHOLD) for dist in dist_range]

# Create the plot
plt.figure(figsize=(10, 6))  # Increase the figure size
plt.plot(dist_range, rd_values, linewidth=2.5)  # Make the curve bolder
plt.axvline(x=THRESHOLD, color='r', linestyle='--', linewidth=2, label='THRESHOLD = 0.1')  # Add a vertical line at THRESHOLD
plt.xlabel('Distance', fontsize=14)  # Increase the font size of x-axis label
plt.ylabel('RD Function Value', fontsize=14)  # Increase the font size of y-axis label
plt.title('RD Function with THRESHOLD = 0.1', fontsize=16)  # Increase the font size of title
plt.legend(fontsize=12)  # Show the legend and increase its font size

# Customize the grid
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)  

# Display the plot
plt.show()
