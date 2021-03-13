import simulator as sim

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

# Spacing of the time steps.
dt = 0.0002
# speed of sound.
c = 10
# Number of grid points.
n = 100
# Number of grid points in x- and y-direction
dim = (n, n)
# Number of time steps.
t = 100
# Grid spacing.
dx = 1 / (n - 1)

# Define the initial conditions
x_coord = np.arange(0., n * dx, dx)

x_mat, y_mat = np.meshgrid(x_coord, x_coord, sparse=True)

# Initial amplitudes.
a0 = np.exp(-((x_mat - 0.5) ** 2 + (y_mat - 0.5) ** 2) / (2 * 0.2 ** 2))
print(a0.shape)
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
my_sim = sim.Numeric2DWaveSimulator(dx, dt, c, dim, t, a0, v0, "fixed edges")
start = time.time()
my_sim.update()
print("shape of the current amplitudes is:", my_sim.current_amplitudes.shape)
print("Shape of the time evoluation matrix is:", my_sim.amplitudes_time_evolution.shape)
result = my_sim.run()
end = time.time()
print("Executing the simulation takes:", "%0.04f" % (end - start), "s")

print(result.shape)
print(result[0, ...].shape)

# my_sim.save_data()

# Here begins the visualization part.
fig, ax = plt.subplots(figsize=(3, 3))
ax.set(xlim=(0, (n - 1) * dx), ylim=(0, (n - 1) * dx))

# contour_opts = {'levels': np.linspace(-9, 9, 10), 'cmap':'RdBu', 'lw': 2}

cax = ax.contour(x_coord, x_coord, result[0, ...])

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
ax.set_title("2D wave equation simulation")


def animate(i):
    ax.collections = []
    ax.contour(x_coord, x_coord, result[i, ...])


# call the animator.  blit=True means only re-draw the parts that have changed.

anim = animation.FuncAnimation(fig, animate, frames=t-1, interval=50)

# anim.save("wave_animation.gif", fps=30)

# print("Image was saved.")

plt.draw()
plt.show()
