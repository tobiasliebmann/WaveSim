import simulator as sim

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

# Spacing of the time steps.
dt = 1
# speed of sound.
c = 1 / np.sqrt(2)
# Number of grid points.
n = 100
m = 100
# Number of grid points in x- and y-direction
dim = (m, n)
# Number of time steps.
t = 100
# Grid spacing.
dx = 1

# Define the initial conditions
x_coord = np.arange(0., n * dx, dx)
y_coord = np.arange(0., n * dx, dx)
x_mat, y_mat = np.meshgrid(x_coord, y_coord, sparse=True)


# Initial amplitudes.
def a0_func(x: np.ndarray, y: np.ndarray, center_x: float, center_y: float, width: float) -> np.ndarray:
    """
    Function returning the initial amplitudes for the wave equation. This is Gaussian bell curve in 2D.
    :param x: x-coordinate values at which the bell curve is examined.
    :param y: x-coordinate values at which the bell curve is examined.
    :param center_x: center x-coordinate of the bell curve.
    :param center_y: center y-coordinate of the bell curve.
    :param width: Width of the bell curve.
    :return: Returns a 2D numpy array respresenting the values of the bell curve at the specified x- and y-coordinates.
    """
    return np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * width ** 2))


# Initial amplitudes.
a0 = a0_func(x_mat, y_mat, (dx * m) / 2., (dx * n) / 2., 10.)

# Initial velocities of the amplitudes.
v0 = np.zeros(dim)

# run the simulation.
my_sim = sim.Numeric2DWaveSimulator(dx, dt, c, dim, t, a0, v0, "fixed edges")
start = time.time()
result = my_sim.run()
end = time.time()
print(f"Executing the simulation takes {round(end-start, 2)} s.")

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
plot = ax.plot_surface(x_mat, y_mat, a0, color='0.75', rstride=1, cstride=1)


def update_plot(frame_number):
    global plot
    plot.remove()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plot = ax.plot_surface(x_mat, y_mat, result[frame_number, ...], cmap="magma")


ax.set_zlim(np.min(result), np.max(result))
ani = animation.FuncAnimation(fig, update_plot, t - 1, interval=100)
print("Starting to save the animation.")
fn = "wave_sim2D_surface4"
ani.save(fn + ".gif", writer="imagemagick", fps=30)
print("Saved the animation.")

plt.draw()
plt.show()
