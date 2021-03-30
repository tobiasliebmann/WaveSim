import simulator as sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

my_sim = sim.Numeric2DWaveSimulator.init_from_file("simulation_test_file.pkl")

result = my_sim.amplitudes_time_evolution

x_coord, y_coord = my_sim.calculate_grid_coordinates()

x_coord = np.reshape(x_coord, (my_sim.number_of_grid_points[0],))
y_coord = np.reshape(y_coord, (my_sim.number_of_grid_points[1],))

result = np.array(result)

fig, ax = plt.subplots(figsize=(5, 5))
ax.set(xlim=(0, (my_sim.number_of_grid_points[0] - 1) * my_sim.delta_x), ylim=(0, (my_sim.number_of_grid_points[1] - 1)
                                                                               * my_sim.delta_x))
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
ax.set_title("2D wave equation simulation")


def animate(i):
    ax.collections = []
    ax.contour(x_coord, y_coord, result[i, ...])
    ax.set_title("Frame:" + str(i))


# call the animator.  blit=True means only re-draw the parts that have changed.

anim = animation.FuncAnimation(fig, animate, frames=my_sim.number_of_time_steps - 1, interval=50)

# anim.save("wave_animation2D.gif", fps=30)

# print("Image was saved.")

plt.draw()
plt.show()
