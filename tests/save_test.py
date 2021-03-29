import simulator as sim

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# todo: Update this to work with functions.
my_sim = sim.Numeric1DWaveSimulator.init_from_file("example_data.npy")
my_sim.run()
# Here begins the visualization part.
fig = plt.figure()
ax = plt.axes(xlim=(0, (my_sim.number_of_grid_points - 1) * my_sim.delta_x), ylim=(-1.5, 1.5))
line, = ax.plot([], [], lw=2)

ax.set_xlabel("position")
ax.set_ylabel("amplitude")
ax.grid(True)
ax.set_title("1D wave equation simulation")

x = np.linspace(0, my_sim.delta_x * my_sim.number_of_grid_points, my_sim.number_of_grid_points)


# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,


# animation function.  This is called sequentially
def animate(i):
    y = my_sim.amplitudes_time_evolution[i]
    line.set_data(x, y)
    return line,


# call the animator.  blit=True means only re-draw the parts that have changed.

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=my_sim.number_of_time_steps, interval=50, blit=True
                               , repeat=True)

# anim.save("wave_animation.gif", fps=30)

# print("Image was saved.")

plt.show()
