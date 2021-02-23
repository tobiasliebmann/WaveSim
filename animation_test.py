import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

TWOPI = 2 * np.pi

fig, ax = plt.subplots()

t = np.arange(0.0, TWOPI, 0.001)
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
