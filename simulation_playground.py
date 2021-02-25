import simulator1D as sim1D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initial amplitudes.
a0 = np.array([0, -1, 1, -1, 1, -1, 1, -1, 1, 0])
# Initial velocities of the amplitudes.
v0 = np.array([0, 0, 1, 1, -1, 0, 0, 0, 0, 0])
# Grid spacing.
dx = 0.1
# Spacing of the time steps.
dt = 0.005
# speed of sound.
c = 10
# Number of grid points.
n = 10
# Number of time steps.
t = 11
my_sim = sim1D.Numeric1DWaveSimulator(dx, dt, c, n, t, a0, v0)
result = my_sim.run()
print(result)

# Here begins the visualization part.

fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(-1.5, 1.5))
line, = ax.plot([], [], lw=2)


# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,


# animation function.  This is called sequentially
def animate(i):
    x = np.linspace(0, dx * n, n)
    y = result[i]
    line.set_data(x, y)
    return line,


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=t, interval=500, blit=True, repeat=True)

plt.show()
