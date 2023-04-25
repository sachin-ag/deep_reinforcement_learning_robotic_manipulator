import sys
import matplotlib.pyplot as plt


def read_coordinates_from_file(file_name):
    coordinates = []
    with open(file_name, 'r') as file:
        for line in file:
            x, y, z = map(float, line.strip().split())
            coordinates.append((x, y, z))
    return coordinates


def plot_3d_scatter(coordinates):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = zip(*coordinates)
    ax.scatter(xs, ys, zs)

    # Draw a curve connecting the points
    ax.plot(xs, ys, zs, color='r', linestyle=':')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    ax.set_zlim(0, 1)  # Set the z-axis limits to 0 and 1

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        file_name = "trajectories/crown"
    coordinates = read_coordinates_from_file(file_name)
    plot_3d_scatter(coordinates)
