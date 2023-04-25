import matplotlib.pyplot as plt
import sys

if len(sys.argv) > 2:
    act_file = sys.argv[1]
    pred_file = sys.argv[2]
else:
    act_file = "trajectories/crown"
    pred_file = "results/crown_pred_path.txt"

# Read files and store coordinates in lists
with open(act_file, "r") as f:
    actual_coordinates = [list(map(float, line.strip().split())) for line in f.readlines()]

with open(pred_file, "r") as f:
    predicted_coordinates = [list(map(float, line.strip().split())) for line in f.readlines()]

# Prepare data for plotting
time = list(range(1, len(actual_coordinates) + 1))
actual_x, actual_y, actual_z = zip(*actual_coordinates)
predicted_x, predicted_y, predicted_z = zip(*predicted_coordinates)

# Calculate errors for each axis
error_x = [abs(a - p) for a, p in zip(actual_x, predicted_x)]
error_y = [abs(a - p) for a, p in zip(actual_y, predicted_y)]
error_z = [abs(a - p) for a, p in zip(actual_z, predicted_z)]

# Function to create a colorful graph
def create_subplot(ax, title, xlabel, ylabel, data, labels, colors):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for d, l, c in zip(data, labels, colors):
        ax.plot(time, d, label=l, color=c)
    ax.legend()

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.tight_layout(pad=5)

# Plot actual vs predicted for each axis
create_subplot(axs[0, 0], "Actual vs Predicted X-axis", "Episode", "X", [actual_x, predicted_x], ["Actual", "Predicted"], ["blue", "orange"])
create_subplot(axs[0, 1], "Actual vs Predicted Y-axis", "Episode", "Y", [actual_y, predicted_y], ["Actual", "Predicted"], ["green", "red"])
create_subplot(axs[0, 2], "Actual vs Predicted Z-axis", "Episode", "Z", [actual_z, predicted_z], ["Actual", "Predicted"], ["purple", "yellow"])

# Plot errors for each axis
create_subplot(axs[1, 0], "Error in X-axis", "Episode", "Error", [error_x], ["Error"], ["blue"])
create_subplot(axs[1, 1], "Error in Y-axis", "Episode", "Error", [error_y], ["Error"], ["green"])
create_subplot(axs[1, 2], "Error in Z-axis", "Episode", "Error", [error_z], ["Error"], ["purple"])

plt.show()
