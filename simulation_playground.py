import simulator1D as sim1D

import visualization as vis

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initial amplitudes.
a0 = np.array([0, -1, 1, -1, 1, -1, 1, -1, 1, 0])
# Initial velocities of the amplitudes.
v0 = np.array([0, 0, 1, 1, -1, 0, 0, 0, 0, 0])
# Grid spacing.
dx: float = 0.1
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
fig, ax = plt.subplots()

t = 
s = np.sin(t)
l = plt.plot(t, s)

ax = plt.axis([0, TWOPI, -1, 1])

redDot, = plt.plot([], [], "ro")


def animate(i):
    redDot.set_data(np.array([i, i + 1]), np.array([np.sin(i), np.sin(i + 1)]))
    return redDot,


# create animation using the animate() function
my_animation = animation.FuncAnimation(fig, animate, frames=np.arange(0.0, TWOPI, 0.1), interval=30, blit=True,
                                       repeat=True)
plt.show()
