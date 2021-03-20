import numpy as np
import random as rd
import matplotlib.pyplot as plt
import time
import numba as nb

dimensions = np.array([10, 50, 100, 200, 300, 400, 500])
dim = 100

times_called = np.array([10, 50, 100, 200, 300, 400, 500])
calls = 100

num = rd.random()

# Arrays saving the data for runs differing how often a function is called but with a fixed dimension.
normal_time_data = np.array([])
opt_time_data = np.array([])
jited_normal_time_data = np.array([])
jited_opt_time_data = np.array([])

# Arrays saving the data for runs differing in dimension but with a fixing the times a function is called.
normal_dim_data = np.array([])
opt_dim_data = np.array([])
jited_normal_dim_data = np.array([])
jited_opt_dim_data = np.array([])


def create_matrix(matrix_dimension):
    """

    :return:
    """
    matrix = (1 - 2 * num) * np.identity(matrix_dimension)
    np.fill_diagonal(matrix[1:], num)
    np.fill_diagonal(matrix[:, 1:], num)
    return matrix


def mat_mul(matrix1, matrix2):
    """

    :param matrix1:
    :param matrix2:
    :return:
    """
    return np.dot(matrix1, matrix2) + np.dot(matrix1, matrix2)


def opt_mat_mul(mat_dim, matrix):
    """

    :param mat_dim:
    :param matrix:
    :return:
    """
    temp1 = np.roll(matrix, mat_dim)
    temp2 = np.roll(matrix, -mat_dim)
    temp3 = np.roll(matrix.T, mat_dim)
    temp4 = np.roll(matrix.T, -mat_dim)
    zeros = np.zeros(mat_dim)
    temp1[0] = zeros
    temp2[mat_dim - 1] = zeros
    temp3[0] = zeros
    temp4[mat_dim - 1] = zeros
    return (2. - 4. * num) * matrix + num * (temp1 + temp2 + temp3.T + temp4.T)


# print(opt_mat_mul() - mat_mul())


def run_mat_mul(matrix1, matrix2, number_of_calls):
    """

    :param matrix1:
    :param matrix2:
    :param number_of_calls:
    :return:
    """
    for _ in range(number_of_calls):
        mat_mul(matrix1, matrix2)


def run_opt_mat_mul(matrix, mat_dim, number_of_calls):
    """

    :param matrix:
    :param mat_dim:
    :param number_of_calls:
    :return:
    """
    for _ in range(number_of_calls):
        opt_mat_mul(mat_dim, matrix)


@nb.jit(nopython=True)
def run_jited_mat_mul(matrix1, matrix2, number_of_calls):
    """

    :param matrix1:
    :param matrix2:
    :param number_of_calls:
    :return:
    """
    for _ in range(number_of_calls):
        return np.dot(matrix1, matrix2) + np.dot(matrix1, matrix2)


@nb.jit(nopython=True)
def run_jited_opt_mat_mul(matrix, mat_dim, number_of_calls):
    """

    :param mat_dim:
    :param matrix:
    :param number_of_calls:
    :return:
    """
    for _ in range(number_of_calls):
        temp1 = np.roll(matrix, mat_dim)
        temp2 = np.roll(matrix, -mat_dim)
        temp3 = np.roll(matrix.T, mat_dim)
        temp4 = np.roll(matrix.T, -mat_dim)
        zeros = np.zeros(mat_dim)
        temp1[0] = zeros
        temp2[mat_dim - 1] = zeros
        temp3[0] = zeros
        temp4[mat_dim - 1] = zeros
        return (2. - 4. * num) * matrix + num * (temp1 + temp2 + temp3.T + temp4.T)


multiply_matrix = create_matrix(dim)
state_matrix = np.random.rand(dim, dim)

for times in times_called:
    start1 = time.time()
    run_jited_opt_mat_mul(state_matrix, dim, times)
    end1 = time.time()
    jited_opt_time_data = np.append(jited_opt_time_data, end1 - start1)

    start2 = time.time()
    run_jited_mat_mul(multiply_matrix, state_matrix, times)
    end2 = time.time()
    jited_normal_time_data = np.append(jited_normal_time_data, end2 - start2)

    start3 = time.time()
    run_mat_mul(multiply_matrix, state_matrix, times)
    end3 = time.time()
    normal_time_data = np.append(normal_time_data, end3 - start3)

    start4 = time.time()
    run_opt_mat_mul(state_matrix, dim, times)
    end4 = time.time()
    opt_time_data = np.append(opt_time_data, end4 - start4)

for dimension in dimensions:

    multiply_matrix = create_matrix(dimension)
    state_matrix = np.random.rand(dimension, dimension)

    start1 = time.time()
    run_jited_opt_mat_mul(state_matrix, dimension, calls)
    end1 = time.time()
    jited_opt_dim_data = np.append(jited_opt_dim_data, end1 - start1)

    start2 = time.time()
    run_jited_mat_mul(multiply_matrix, state_matrix, calls)
    end2 = time.time()
    jited_normal_dim_data = np.append(jited_normal_dim_data, end2 - start2)

    start3 = time.time()
    run_mat_mul(multiply_matrix, state_matrix, calls)
    end3 = time.time()
    normal_dim_data = np.append(normal_dim_data, end3 - start3)

    start4 = time.time()
    run_opt_mat_mul(state_matrix, dimension, calls)
    end4 = time.time()
    opt_dim_data = np.append(opt_dim_data, end4 - start4)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10., 4.8))

ax1.plot(times_called, normal_time_data, marker="s")
ax1.plot(times_called, opt_time_data, marker="o")
ax1.plot(times_called, jited_normal_time_data, marker="d")
ax1.plot(times_called, jited_opt_time_data, marker="*")

ax1.set_ylabel("execution time (s)")
ax1.set_xlabel("times called")
ax1.set_yscale("log")
ax1.set_title(f"Fixed matrix dimensions: {dim}x{dim}")
ax1.legend(["Normal", "Optimized", "Jited and normal", "Jited and optimized"], loc="upper left", bbox_to_anchor=(0.075, .995))

ax2.plot(dimensions, normal_dim_data, marker="s")
ax2.plot(dimensions, opt_dim_data, marker="o")
ax2.plot(dimensions, jited_normal_dim_data, marker="d")
ax2.plot(dimensions, jited_opt_dim_data, marker="*")

ax2.set_xlabel("matrix dimensions")
ax2.set_yscale("log")
ax2.set_title(f"Fixed number of calls: {calls}")

ax1.grid(True)
ax2.grid(True)
plt.tight_layout()
plt.draw()
plt.show()
