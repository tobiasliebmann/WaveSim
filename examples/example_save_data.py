import simulator as sim
import numpy as np
import time

# Spacing of the time steps.
dt = 1.
# speed of sound.
c = 1 / np.sqrt(2)
# Number of grid points.
n = 200
m = 200
# Number of grid points in x- and y-direction
dim = (n, m)
# Number of time steps.
t = 500
# Grid spacing.
dx = 1.

# Define the initial conditions
x_coord = np.arange(0., n * dx, dx)
y_coord = np.arange(0., m * dx, dx)
# x_mat, y_mat = np.meshgrid(x_coord, y_coord, sparse=True)


# Initial amplitudes.
def amp_func(x, y):
    def a0_func(x: np.ndarray, y: np.ndarray, center_x: float, center_y: float, width: float) -> np.ndarray:
        """
        Function returning the initial amplitudes for the wave equation. This is Gaussian bell curve in 2D.
        :param x: x-coordinate values at which the bell curve is examined.
        :param y: x-coordinate values at which the bell curve is examined.
        :param center_x: center x-coordinate of the bell curve.
        :param center_y: center y-coordinate of the bell curve.
        :param width: Width of the bell curve.
        :return: Returns a 2D numpy array respresenting the values of the bell curve at the specified x- and y-coordinates.
        """
        return np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * width ** 2))
    return a0_func(x, y, 100., 100., 2.)


def vel_func(x, y):
    return 0. * x + 0. * y


# run the simulation.
my_sim = sim.Numeric2DWaveSimulator(dx, dt, c, dim, t, amp_func, vel_func, "loose edges")
start = time.time()
result = my_sim.run()
end = time.time()
print(f"Executing the simulation takes {round(end - start, 2)} s.")

my_sim.save_data(link_to_file="simulation_test_file.pkl")
