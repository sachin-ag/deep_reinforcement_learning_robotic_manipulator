import numpy as np
import matplotlib.pyplot as plt

# Read the accuracy values from file
with open('accuracy.txt', 'r') as f:
    accuracy_values = [float(line.strip()) for line in f]
    print(accuracy_values)

# Calculate the 1000 moving average
moving_avg = np.convolve(accuracy_values, np.ones(100)/100, mode='valid')

# Calculate the standard deviation
std = .25*np.std(moving_avg)

# Create the plot
fig, ax = plt.subplots()

# Plot the 1000 moving average as a line
ax.plot(range(len(moving_avg)), moving_avg, label='Moving Average')

# Add shaded region for the standard deviation
ax.fill_between(range(len(moving_avg)), moving_avg-std, moving_avg+std, alpha=0.2, label='Standard Deviation')

# Add axis labels and legend
ax.set_xlabel('Episode Number')
ax.set_ylabel('Accuracy')
ax.legend()

# Show the plot
plt.show()
