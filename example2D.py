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
m = 100
# Number of grid points in x- and y-direction
dim = (m, n)
# Number of time steps.
t = 300
# Grid spacing.
dx = 1

# Define the initial conditions
x_coord = np.arange(0., n * dx, dx)
y_coord = np.arange(0., m * dx, dx)
x_mat, y_mat = np.meshgrid(x_coord, y_coord, sparse=True)
# Initial amplitudes.
a0 = np.exp(-((x_mat - (dx * m)/2) ** 2 + (y_mat - (dx * n)/2) ** 2) / (2 * 2. ** 2))
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
my_sim = sim.Numeric2DWaveSimulator(dx, dt, c, dim, t, a0, v0, "loose edges")
start = time.time()
result = my_sim.run()
end = time.time()
print("Executing the simulation takes:", "%0.04f" % (end - start), "s")

with np.printoptions(precision=5, suppress=True):
    print(my_sim.time_step_matrix_left,"\n")
    print(my_sim.time_step_matrix_right)
    print(result[0])
    print(result[t-1])

fig, ax = plt.subplots(figsize=(3, 3))
ax.set(xlim=(0, (n - 1) * dx), ylim=(0, (n - 1) * dx))

contour_opts = {"levels": np.linspace(-9, 9, 10), "cmap": "RdBu", "lw": 2}

cax = ax.contour(x_coord, x_coord, a0)

# plt.draw()
# plt.show()

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
ax.set_title("2D wave equation simulation")


def animate(i):
    ax.collections = []
    ax.contour(x_coord, x_coord, result[i, ...])
    ax.set_title("Frame:"+str(i))


# call the animator.  blit=True means only re-draw the parts that have changed.

anim = animation.FuncAnimation(fig, animate, frames=t-1, interval=50)

# anim.save("wave_animation.gif", fps=30)

# print("Image was saved.")

plt.draw()
plt.show()
