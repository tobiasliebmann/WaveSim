import numpy as np
import random as rd
import timeit as ti
import time
import numba as nb

dim = 100

num = rd.random()


def create_matrix(dim, num):
    """

    :param num:
    :param dim:
    :return:
    """
    matrix = (1 - 2 * num) * np.identity(dim)
    np.fill_diagonal(matrix[1:], num)
    np.fill_diagonal(matrix[:, 1:], num)
    return matrix


multiply_matrix = create_matrix(dim, num)
state_matrix = np.random.rand(dim, dim)


def mat_mul():
    return np.matmul(multiply_matrix, state_matrix)


@nb.jit(nopython=True, parallel=True)
def opt_mat_mul():
    temp1 = np.roll(state_matrix, dim)
    temp2 = np.roll(state_matrix, -dim)
    zeros = np.zeros(dim)
    temp1[0] = zeros
    temp2[dim - 1] = zeros
    return (1 - 2 * num) * state_matrix + num * (temp1 + temp2)


opt_mat_mul()

start1 = time.time()
[mat_mul() for i in range(100)]
end1 = time.time()

start2 = time.time()
[opt_mat_mul() for j in range(100)]
end2 = time.time()

# print(f"Function call needed {ti.timeit(mat_mul)} ms.")
# print(f"Optimized function call needed {ti.timeit(opt_mat_mul)} ms.")
print((end1 - start1))
print((end2 - start2))
