import simulator as sim

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

# Spacing of the time steps.
dt = 1
# speed of sound.
c = 1/np.sqrt(2)
# Number of grid points.
n = 100
m = 150
# Number of grid points in x- and y-direction
dim = (n, m)
# Number of time steps.
t = 500
# Grid spacing.
dx = 1

# Define the initial conditions
x_coord = np.arange(0., n * dx, dx)
y_coord = np.arange(0., m * dx, dx)
x_mat, y_mat = np.meshgrid(x_coord, y_coord, sparse=True)


# Initial amplitudes.
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


def amp_func(x, y):
    global dx
    global m
    global n
    return a0_func(x, y, (dx * m)/2., (dx * n)/2., 2.)


def vel_func(x, y):
    return 0.*x + 0.*y


a0 = a0_func(x_mat, y_mat, (dx * n)/2., (dx * m)/2., 2.)

# a0 = np.exp(-((x_mat - (dx * m)/2) ** 2 + (y_mat - (dx * n)/2) ** 2) / (2 * 2. ** 2))
# a0 = np.cos(x_coord)
# a0[0] = 0.
# a0[-1] = 0.

# Initial velocities of the amplitudes.
v0 = np.zeros(dim)
# v0 = np.cos(x_coord)
# v0[0] = 0.
# v0[-1] = 0.
# print(v0)

# run the simulation.
my_sim = sim.Numeric2DWaveSimulator(dx, dt, c, dim, t, amp_func, vel_func, "loose edges")
start = time.time()
result = my_sim.run()
end = time.time()
print(f"Executing the simulation takes {round(end-start, 2)} s.")

result = np.array(result)

print(result.shape)

fig, ax = plt.subplots(figsize=(5, 5))
ax.set(xlim=(0, (m - 1) * dx), ylim=(0, (n - 1) * dx))

# contour_opts = {"levels": np.linspace(-9, 9, 10), "cmap": "RdBu", "lw": 2}

cax = ax.contour(x_coord, y_coord, a0)

# plt.draw()
# plt.show()

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
ax.set_title("2D wave equation simulation")


def animate(i):
    ax.collections = []
    ax.contour(y_coord, x_coord, result[i, ...])
    ax.set_title("Frame:"+str(i))


# call the animator.  blit=True means only re-draw the parts that have changed.

anim = animation.FuncAnimation(fig, animate, frames=t-1, interval=50)

# anim.save("wave_animation2D.gif", fps=30)

# print("Image was saved.")

plt.draw()
plt.show()
