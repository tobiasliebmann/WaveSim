import numpy as np
import matplotlib.pyplot as plt
import time as tm
import simulator as sm

t_start = 100
call_times = [100, 150, 300, 500, 750, 1000]

dim_1D_start = 100
dims_1D = [100, 150, 300, 500, 750, 1000]

dim_2D_start = (100, 100)
dims_2D = [(100, 100), (150, 150), (300, 300), (500, 500), (750, 750), (1000, 1000)]

dx = 1.
dt = 1.
c = 1/np.sqrt(2)


def init_amp_1d(x: np.ndarray) -> np.ndarray:
    return np.random.rand(*x.shape)


def init_amp_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.5 * (np.random.rand(*x.shape) + np.random.rand(*y.shape))


def init_vel_1d(x: np.ndarray) -> np.ndarray:
    return 0. * x


def init_vel_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0. * x + 0. * y


my_sim1D = sm.Numeric1DWaveSimulator(dx, dt, c, dim_1D_start, t_start, init_amp_1d, init_vel_1d)
my_sim2d = sm.Numeric2DWaveSimulator(dx, dt, c, dim_2D_start, t_start, init_amp_2d, init_vel_2d, "fixed edges")


def time_dim(sim: sm.NumericWaveSimulator, new_dim):
    sim.number_of_grid_points = new_dim
    start = tm.time()
    sim.run()
    end = tm.time()
    return end - start


def time_calls(sim: sm.NumericWaveSimulator, number_of_calls):
    # print(number_of_calls)
    sim.number_of_time_steps = number_of_calls
    start = tm.time()
    sim.run()
    end = tm.time()
    return end - start


dim_time_data_1d = [time_dim(my_sim1D, dimension) for dimension in dims_1D]
dim_time_data_2d = [time_dim(my_sim2d, dimension) for dimension in dims_2D]

my_sim1D.number_of_grid_points = dim_1D_start
my_sim2d.number_of_grid_points = dim_2D_start

calls_time_data_1d = [time_calls(my_sim1D, calls) for calls in call_times]
calls_time_data_2d = [time_calls(my_sim2d, calls) for calls in call_times]

# --------------------
# Plotting the results
# --------------------

# Axes and fiure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10., 4.8))

# Graphs for the first plot.
ax1.plot(dims_1D, dim_time_data_1d, marker="s")
ax1.plot(dims_1D, dim_time_data_2d, marker="o")

# Labels ad legends for the first plot
ax1.set_ylabel("execution time (s)")
ax1.set_xlabel("number of function calls")
ax1.set_yscale("log")
ax1.set_title(f"Fixed dimension: {dim_1D_start}")
ax1.legend(["1D", "2D"], loc="upper left", bbox_to_anchor=(0.0, 1.01))

# Graphs for the second plot
ax2.plot(call_times, calls_time_data_1d, marker="s")
ax2.plot(call_times, calls_time_data_2d, marker="o")

# Labels and legends for th second plot
ax2.set_xlabel("dimension")
ax2.set_yscale("log")
ax2.set_title(f"Fixed number of calls: {t_start}")

# Show everything
ax1.grid(True)
ax2.grid(True)
plt.tight_layout()
plt.draw()
plt.show()

