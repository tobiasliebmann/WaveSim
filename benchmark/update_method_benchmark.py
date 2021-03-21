import numpy as np

import numba as nb

import time as tm

# Spacing of the time steps.
dt = 1
# speed of sound.
c = 1 / np.sqrt(2)
# Number of grid points.
n = 300
m = n
# Number of grid points in x- and y-direction
dim = (m, n)
# Number of time steps.
t = 100
# Grid spacing.
dx = 1

num = (c * dt / dx) ** 2

# Define the initial conditions
x_coord = np.arange(0., n * dx, dx)
y_coord = np.arange(0., m * dx, dx)
x_mat, y_mat = np.meshgrid(x_coord, y_coord, sparse=True)


# Initial amplitudes.
# Initial amplitudes.
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


a0 = a0_func(x_mat, y_mat, (dx * m) / 2., (dx * n) / 2., 2.)

v0 = np.zeros(dim)


def create_matrix(number: float, matrix_dimension: int) -> np.ndarray:
    """
    This function creates a quadratic matrix with a populated diagonal, populated off-diagonals and dimension
    matrix_dimension. All entries in the diagonal are (1 - 2 * number) while the entreis in the off-diagonal are the
    number number.
    :param number: A float number which populates the off-diagonals of the matrix.
    :param matrix_dimension: Dimension of the quadratic matrix which is returned.
    :return: The matrix constructed as described above.
    """
    matrix = (1 - 2 * number) * np.identity(matrix_dimension)
    np.fill_diagonal(matrix[1:], number)
    np.fill_diagonal(matrix[:, 1:], number)
    return matrix


left_matrix = create_matrix(num, n)
right_matrix = left_matrix


@nb.jit(nopython=True)
def jit_cal_amp(nots: int, deltat: float, left_mat: np.ndarray, right_mat: np.ndarray, init_amp: np.ndarray,
                init_vel: np.ndarray):
    """

    :return:
    """
    time_evo_stack = init_amp
    # The first is given by this equation.
    former_amp = init_amp
    curr_amp = 0.5 * (np.dot(left_mat, init_amp) + np.dot(init_amp, right_mat)) + deltat * init_vel
    time_evo_stack = np.append(time_evo_stack, curr_amp)

    for _ in range(nots - 2):
        temp = curr_amp
        curr_amp = np.dot(left_mat, curr_amp) + np.dot(curr_amp, right_mat) - former_amp
        former_amp = temp
        time_evo_stack = np.append(time_evo_stack, curr_amp)

    return time_evo_stack


def cal_amp(nots: int, deltat: float, left_mat: np.ndarray, right_mat: np.ndarray, init_amp: np.ndarray,
            init_vel: np.ndarray):
    """

    :return:
    """
    time_evo_stack = init_amp
    # The first is given by this equation.
    former_amp = init_amp
    curr_amp = 0.5 * (np.dot(left_mat, init_amp) + np.dot(init_amp, right_mat)) + deltat * init_vel
    time_evo_stack = np.append(time_evo_stack, curr_amp)

    for _ in range(nots - 2):
        temp = curr_amp
        curr_amp = np.dot(left_mat, curr_amp) + np.dot(curr_amp, right_mat) - former_amp
        former_amp = temp
        time_evo_stack = np.append(time_evo_stack, curr_amp)

    return time_evo_stack


jit_cal_amp(t, dt, left_matrix, right_matrix, a0, v0)

start1 = tm.time()
cal_amp(t, dt, left_matrix, right_matrix, a0, v0)
end1 = tm.time()

start2 = tm.time()
jit_cal_amp(t, dt, left_matrix, right_matrix, a0, v0)
end2 = tm.time()

print(f"The jited function call took {end2 - start2} s.")
print(f"The normal function call took {end1 - start1} s.")
