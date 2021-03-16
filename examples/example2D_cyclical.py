import simulator as sim

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

# Spacing of the time steps.
dt = float(1)
# speed of sound.
c = 1/np.sqrt(2)
# Number of grid points.
n = 100
m = 100
# Number of grid points in x- and y-direction
dim = (m, n)
# Number of time steps.
t = 300
# Grid spacing.
dx = float(1)

# Define the initial conditions
x_coord = np.arange(0., n * dx, dx)
y_coord = np.zeros(n)
x_mat, y_mat = np.meshgrid(x_coord, y_coord, sparse=True)


# Initial amplitudes.
def a0_func(x, y, center_x, center_y, width):
    """

    :param x:
    :param y:
    :param center_x:
    :param center_y:
    :param width:
    :return:
    """
    return np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * width ** 2))


a0 = a0_func(x_mat, y_mat, (dx * m)/3., 0., 2.)

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
my_sim = sim.Numeric2DWaveSimulator(dx, dt, c, dim, t, a0, v0, "cyclical")
start = time.time()
result = my_sim.run()
end = time.time()
print(f"Executing the simulation takes {round(end-start, 2)} s.")

fig, ax = plt.subplots(figsize=(5, 5))
ax.set(xlim=(0, (n - 1) * dx), ylim=(0, (n - 1) * dx))

# contour_opts = {"levels": np.linspace(-9, 9, 10), "cmap": "RdBu", "lw": 2}

cax = ax.contourf(x_coord, x_coord, a0)

# plt.draw()
# plt.show()

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
ax.set_title("2D wave equation simulation")


def animate(i):
    ax.collections = []
    ax.contourf(x_coord, x_coord, result[i, ...])
    ax.set_title("time:"+str(round(float(i*dt), 2)))


# call the animator.  blit=True means only re-draw the parts that have changed.

anim = animation.FuncAnimation(fig, animate, frames=t-1, interval=50)

# anim.save("wave_animation_2D.gif", fps=30)

# print("Image was saved.")

plt.draw()
plt.show()
