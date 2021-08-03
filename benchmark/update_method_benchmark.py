import numpy as np
from scipy import sparse as sp
import time as tm
import numba as nb


# Spacing of the time steps.
dt = 1
# speed of sound.
c = 1 / np.sqrt(3)
# Number of grid points.
n = 100
m = n
# Number of grid points in x- and y-direction
dim = (m, n)
# Number of time steps.
t = 500
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
    :return: Returns a 2D numpy array representing the values of the bell curve at the specified x- and y-coordinates.
    """
    return np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * width ** 2))


a0 = a0_func(x_mat, y_mat, (dx * m) / 2., (dx * n) / 2., 2.)

v0 = np.zeros(dim)


def create_matrix(number: float, matrix_dimension: int) -> np.ndarray:
    """
    This function creates a quadratic matrix with a populated diagonal, populated off-diagonals and dimension
    matrix_dimension. All entries in the diagonal are (1 - 2 * number) while the entries in the off-diagonal are the
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

left_matrix_sparse = sp.dia_matrix(left_matrix)
right_matrix_sparse = sp.dia_matrix(right_matrix)


def cal_amp_sparse(number_of_calls: int, left_mat: sp.dia_matrix, right_mat: sp.dia_matrix, init_amp: np.ndarray):
    """

    :return:
    """
    return [left_mat.dot(init_amp) - init_amp
            for _ in range(number_of_calls)]


def cal_amp(number_of_calls: int, left_mat: np.ndarray, right_mat: np.ndarray, init_amp: np.ndarray):
    """

    :return:
    """
    return [np.dot(left_mat, init_amp) - init_amp
            for _ in range(number_of_calls)]


@nb.jit(nopython=True)
def jitted_sim(dimension: tuple, delta_t: float, number_of_time_steps: int, matrix_l: np.ndarray, matrix_r: np.ndarray,
               init_amps: np.ndarray, init_val: np.ndarray):
    """

    :param dimension:
    :param number_of_time_steps:
    :param delta_t:
    :param matrix_r:
    :param matrix_l:
    :param init_amps:
    :param init_val:
    :return:
    """
    time_evo_stack = np.zeros(dimension, dtype="float64")
    prev_amps = init_amps
    curr_amps = 0.5 * (np.dot(matrix_l, init_amps) + np.dot(init_amps, matrix_r)) + delta_t * init_val
    for i in range(number_of_time_steps):
        temp = curr_amps
        curr_amps = np.dot(matrix_l, curr_amps) + np.dot(curr_amps, matrix_r) - prev_amps
        prev_amps = temp
        time_evo_stack[i] = curr_amps
    return time_evo_stack


start_jit = tm.time()
jitted_sim((t, m, n), dt, t, left_matrix, right_matrix, a0, v0)
end_jit = tm.time()
print(f"The first jitted execution took {end_jit - start_jit} s.")

start_jit2 = tm.time()
jitted_sim((t, m, n), dt, t, left_matrix, right_matrix, a0, v0)
end_jit2 = tm.time()
print(f"The second jitted execution took {end_jit2 - start_jit2} s.")

start1 = tm.time()
cal_amp(t, left_matrix, right_matrix, a0)
end1 = tm.time()

print(f"The repeated normal function call took {end1 - start1} s.")

start2 = tm.time()
cal_amp_sparse(t, left_matrix_sparse, right_matrix_sparse, a0)
end2 = tm.time()

print(f"The repeated function call using sparse matrices took {end2 - start2} s.")

start3 = tm.time()
mat = np.dot(left_matrix, a0)
mat + mat.T
end3 = tm.time()

print(f"The function call using sparse matrices took {end3 - start3} s.")

start4 = tm.time()
np.dot(left_matrix, a0) + np.dot(a0, right_matrix)
end4 = tm.time()

print(f"The function using sparse matrices call took {end4 - start4} s.")
