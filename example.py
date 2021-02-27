import simulator1D as sim1D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

# Spacing of the time steps.
dt = 0.0002
# speed of sound.
c = 10
# Number of grid points.
n = 500
# Number of time steps.
t = 100
# Grid spacing.
dx = 1 / (n - 1)

# Define the initial conditions
x_coord = np.arange(0., n * dx, dx)

# Initial amplitudes.
a0 = np.exp(-(x_coord - 0.5)**2 / (2 * 0.01**2))
# a0 = np.cos(x_coord)
a0[0] = 0.
a0[-1] = 0.

# Initial velocities of the amplitudes.
v0 = np.zeros(n)
# v0 = np.cos(x_coord)
v0[0] = 0.
v0[-1] = 0.
# print(v0)

# run the simulation.
my_sim = sim1D.Numeric1DWaveSimulator(dx, dt, c, n, t, a0, v0)
start = time.time()
result = my_sim.run()
end = time.time()
print("Executing the simulation takes:", "%0.04f" % (end - start), "ms")

# Here begins the visualization part.
fig = plt.figure()
ax = plt.axes(xlim=(0, (n-1)*dx), ylim=(-1.5, 1.5))
line, = ax.plot([], [], lw=2)

ax.set_xlabel("position")
ax.set_ylabel("amplitude")
ax.grid(True)
ax.set_title("1D wave equation simulation")

x = np.linspace(0, dx * n, n)


# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,


# animation function.  This is called sequentially
def animate(i):
    y = result[i]
    line.set_data(x, y)
    return line,


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=t, interval=50, blit=True, repeat=True)

# anim.save("wave_animation.gif", fps=30)

# print("Image was saved.")

plt.show()
