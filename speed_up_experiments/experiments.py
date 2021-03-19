import numpy as np
import random as rd
# import timeit as ti
import time
import numba as nb

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


def mat_mul(matrix1, matrix2):
    """

    :param matrix1:
    :param matrix2:
    :return:
    """
    return np.dot(matrix1, matrix2) + np.dot(matrix1, matrix2)


def opt_mat_mul(matrix):
    """

    :param matrix:
    :return:
    """
    temp1 = np.roll(matrix, dim)
    temp2 = np.roll(matrix, -dim)
    temp3 = np.roll(matrix.T, dim)
    temp4 = np.roll(matrix.T, -dim)
    zeros = np.zeros(dim)
    temp1[0] = zeros
    temp2[dim - 1] = zeros
    temp3[0] = zeros
    temp4[dim - 1] = zeros
    return (2. - 4. * num) * matrix + num * (temp1 + temp2 + temp3.T + temp4.T)


# print(opt_mat_mul() - mat_mul())


def run_mat_mul(times):
    """

    :param times:
    :return:
    """
    for _ in range(times):
        mat_mul(multiply_matrix, state_matrix)


def run_opt_mat_mul(times):
    """

    :param times:
    :return:
    """
    for _ in range(times):
        opt_mat_mul(state_matrix)


@nb.jit(nopython=True)
def run_jited_mat_mul(matrix1, matrix2, times):
    """

    :param matrix1:
    :param matrix2:
    :param times:
    :return:
    """
    for _ in range(times):
        return np.dot(matrix1, matrix2) + np.dot(matrix1, matrix2)


@nb.jit(nopython=True)
def run_jited_opt_mat_mul(matrix, times):
    """

    :param matrix:
    :param times:
    :return:
    """
    for _ in range(times):
        temp1 = np.roll(matrix, dim)
        temp2 = np.roll(matrix, -dim)
        temp3 = np.roll(matrix.T, dim)
        temp4 = np.roll(matrix.T, -dim)
        zeros = np.zeros(dim)
        temp1[0] = zeros
        temp2[dim - 1] = zeros
        temp3[0] = zeros
        temp4[dim - 1] = zeros
        return (2. - 4. * num) * matrix + num * (temp1 + temp2 + temp3.T + temp4.T)


start1 = time.time()
run_jited_opt_mat_mul(state_matrix, times_called)
end1 = time.time()

start2 = time.time()
run_jited_mat_mul(multiply_matrix, state_matrix, times_called)
end2 = time.time()

start3 = time.time()
run_jited_mat_mul(multiply_matrix, state_matrix, times_called)
end3 = time.time()

start4 = time.time()
run_jited_opt_mat_mul(state_matrix, times_called)
end4 = time.time()

start5 = time.time()
run_mat_mul(times_called)
end5 = time.time()

start6 = time.time()
run_opt_mat_mul(times_called)
end6 = time.time()

# print(f"Function call needed {ti.timeit(mat_mul)} ms.")
# print(f"Optimized function call needed {ti.timeit(opt_mat_mul)} ms.")
print(f"Parameters: matrix dimension = {dim}x{dim}, function calls = {times_called}.")
print(f"For the first time the jited standard multiplication takes {np.round(end1 - start1, 4)} s.")
print(f"For the first time the jited optimized multiplication takes {np.round(end2 - start2, 4)} s.")
print(f"For the second time the jited standard multiplication takes {np.round(end3 - start3, 4)} s.")
print(f"For the second time the jited optimized multiplication takes {np.round(end4 - start4, 4)} s.")
print(f"The standard multiplication takes {np.round(end5 - start5, 2)} s.")
print(f"The optimized multiplication takes {np.round(end6 - start6, 2)} s.")
