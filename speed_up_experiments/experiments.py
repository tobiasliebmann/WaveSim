import numpy as np
import random as rd
import timeit as ti
import time
import numba as nb
import scipy

dim = 500

times_called = 300

num = rd.random()


def create_matrix():
    """

    :return:
    """
    matrix = (1 - 2 * num) * np.identity(dim)
    np.fill_diagonal(matrix[1:], num)
    np.fill_diagonal(matrix[:, 1:], num)
    return matrix


multiply_matrix = create_matrix()
state_matrix = np.random.rand(dim, dim)

print(multiply_matrix.dtype)
print(state_matrix.dtype)


def mat_mul():
    return np.dot(multiply_matrix, state_matrix) + np.dot(state_matrix, multiply_matrix)


def opt_mat_mul():
    temp1 = np.roll(state_matrix, dim)
    temp2 = np.roll(state_matrix, -dim)
    temp3 = np.roll(state_matrix.T, dim)
    temp4 = np.roll(state_matrix.T, -dim)
    zeros = np.zeros(dim)
    temp1[0] = zeros
    temp2[dim - 1] = zeros
    temp3[0] = zeros
    temp4[dim - 1] = zeros
    return (2. - 4. * num) * state_matrix + num * (temp1 + temp2 + temp3.T + temp4.T)


# print(opt_mat_mul() - mat_mul())

mat_mul()

start2 = time.time()
[opt_mat_mul() for _ in range(times_called)]
end2 = time.time()

start1 = time.time()
[mat_mul() for _ in range(times_called)]
end1 = time.time()

# print(f"Function call needed {ti.timeit(mat_mul)} ms.")
# print(f"Optimized function call needed {ti.timeit(opt_mat_mul)} ms.")
print((end1 - start1))
print((end2 - start2))
