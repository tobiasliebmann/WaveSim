import simulator as sim

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

# Spacing of the time steps.
dt = 1.
# speed of sound.
c = 1/np.sqrt(2)
# Number of grid points.
n = 100
# Number of time steps.
t = 1000
# Grid spacing.
dx = 1.

# Define the initial conditions
x_coord = np.arange(0., n * dx, dx)


def init_amp_func(x_array: np.ndarray) -> np.ndarray:
    global n
    global dx
    return np.exp(-(x_array - (n * dx)/2) ** 2 / (2 * 10. ** 2))


def init_vel_func(x_array: np.ndarray) -> np.ndarray:
    return 0. * x_array


# run the simulation.
my_sim = sim.Numeric1DWaveSimulator(dx, dt, c, n, t, init_amp_func, init_vel_func)
start = time.time()
result = my_sim.run()
end = time.time()
print(f"Executing the simulation takes {round(end-start, 3)} s.")

my_sim.save_data("example_data.npy")

# Here begins the visualization part.
fig = plt.figure()
ax = plt.axes(xlim=(0, (n - 1) * dx), ylim=(-1.5, 1.5))
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
