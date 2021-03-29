import numpy as np
import random as rd
import matplotlib.pyplot as plt
import time
import numba as nb
from scipy import sparse as sp


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


def mat_mul(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """
    Calculates the anti-commutator of two matrices.
    :param matrix1: A matrix.
    :param matrix2: Another matrix.
    :return: The Anti-commutator matrix1 and matrix2.
    """
    return np.dot(matrix1, matrix2) + np.dot(matrix1, matrix2)


def opt_mat_mul(number: float, matrix_dimension: int, matrix: np.ndarray) -> np.ndarray:
    """
    This function is supposed to be an optimized version of the anti commutator with a matrix that is created by the
    create_matrix function.
    :param number: Float number for the created matrix.
    :param matrix_dimension: Dimension of the created matrix.
    :param matrix: Matrix with which the matrix which is otherwise created by create_matrix is multiplied
    :return: The Anti-commutator.
    """
    temp1 = np.roll(matrix, matrix_dimension)
    temp2 = np.roll(matrix, -matrix_dimension)
    temp3 = np.roll(matrix.T, matrix_dimension)
    temp4 = np.roll(matrix.T, -matrix_dimension)
    zeros = np.zeros(matrix_dimension)
    temp1[0] = zeros
    temp2[matrix_dimension - 1] = zeros
    temp3[0] = zeros
    temp4[matrix_dimension - 1] = zeros
    return (2. - 4. * number) * matrix + number * (temp1 + temp2 + temp3.T + temp4.T)


# print(opt_mat_mul() - mat_mul())


def run_sparse_mat_mul(matrix1: sp.csr_matrix, matrix2: np.ndarray, number_of_calls: int):
    """

    :param number_of_calls:
    :param matrix1:
    :param matrix2:
    :return:
    """
    for _ in range(number_of_calls):
        matrix1.dot(matrix2)


def run_mat_mul(matrix1: np.ndarray, matrix2: np.ndarray, number_of_calls: int):
    """
    Runs the mat_mul function with matrix1 and matrix2 number_of_calls times.
    :param matrix1: A matrix.
    :param matrix2: Another matrix.
    :param number_of_calls: How often to call the mat_mul function
    :return: None.
    """
    for _ in range(number_of_calls):
        mat_mul(matrix1, matrix2)


def run_opt_mat_mul(number: float, matrix: np.ndarray, mat_dim: int, number_of_calls: int):
    """
    Runs the opt_mat_mul function number_of_calls times. Matrix is the matrix to multiplied with the dimension mat_dim.
    :param number: Equivalent of the number used in the create_matrix function.
    :param matrix: Matrix to be multiplied.
    :param mat_dim: Dimension of the matrix.
    :param number_of_calls: How often the opt_mat_mal function is called.
    :return: None
    """
    for _ in range(number_of_calls):
        opt_mat_mul(number, mat_dim, matrix)


@nb.jit(nopython=True)
def run_jited_mat_mul(matrix1: np.ndarray, matrix2: np.ndarray, number_of_calls: int) -> np.ndarray:
    """
    Jited version of the run_mat_mul function.
    """
    for _ in range(number_of_calls):
        return np.dot(matrix1, matrix2) + np.dot(matrix1, matrix2)


@nb.jit(nopython=True)
def run_jited_opt_mat_mul(matrix: np.ndarray, mat_dim: int, number: float, number_of_calls: int) -> np.ndarray:
    """
    JIted version of the run_opt_mat_mul function.
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
        return (2. - 4. * number) * matrix + number * (temp1 + temp2 + temp3.T + temp4.T)


# Dimensions used in the dimension benchmark.
dimensions = np.array([10, 50, 100, 200, 300, 400, 500])

# Fixed dimension for the number of calls benchmark.
dim = 100

# Number of calls used in the benchmark for the number of calls.
times_called = np.array([10, 50, 100, 200, 300, 400, 500])

# Fixed number of calls for the dimension benchmark.
calls = 100

# Arrays saving the data for runs differing how often a function is called but with a fixed dimension.
normal_time_data = np.array([])
opt_time_data = np.array([])
jited_normal_time_data = np.array([])
jited_opt_time_data = np.array([])
sparse_time_data = np.array([])

# Arrays saving the data for runs differing in dimension but with a fixing the times a function is called.
normal_dim_data = np.array([])
opt_dim_data = np.array([])
jited_normal_dim_data = np.array([])
jited_opt_dim_data = np.array([])
sparse_dim_data = np.array([])

# Create a random number and two matrices accordingly.
num = rd.random()
multiply_matrix = create_matrix(num, dim)
state_matrix = np.random.rand(dim, dim)
sparse_matrix = sp.csr_matrix(multiply_matrix)

# Run the following loops for the benchmark.
# The code could probably be a lot cleaner please don't hate me for this.
for times in times_called:
    start1 = time.time()
    run_jited_opt_mat_mul(state_matrix, dim, num, times)
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
    run_opt_mat_mul(num, state_matrix, dim, times)
    end4 = time.time()
    opt_time_data = np.append(opt_time_data, end4 - start4)

    start5 = time.time()
    run_sparse_mat_mul(sparse_matrix, state_matrix, times)
    end5 = time.time()
    sparse_time_data = np.append(sparse_time_data, end5 - start5)

for dimension in dimensions:

    multiply_matrix = create_matrix(num, dimension)
    state_matrix = np.random.rand(dimension, dimension)
    sparse_matrix = sp.csr_matrix(multiply_matrix)

    start1 = time.time()
    run_jited_opt_mat_mul(state_matrix, dimension, num, calls)
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
    run_opt_mat_mul(num, state_matrix, dimension, calls)
    end4 = time.time()
    opt_dim_data = np.append(opt_dim_data, end4 - start4)

    start5 = time.time()
    run_sparse_mat_mul(sparse_matrix, state_matrix, calls)
    end5 = time.time()
    sparse_dim_data = np.append(sparse_dim_data, end5 - start5)

# --------------------
# Plotting the results
# --------------------

# Axes and fiure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10., 4.8))

# Graphs for the first plot.
ax1.plot(times_called, normal_time_data, marker="s")
ax1.plot(times_called, opt_time_data, marker="o")
ax1.plot(times_called, jited_normal_time_data, marker="d")
ax1.plot(times_called, jited_opt_time_data, marker="*")
ax1.plot(times_called, sparse_time_data, marker=".")

# Labels ad legends for the first plot
ax1.set_ylabel("execution time (s)")
ax1.set_xlabel("function calls")
ax1.set_yscale("log")
ax1.set_title(f"Fixed matrix dimensions: {dim}x{dim}")
ax1.legend(["Normal", "Optimized", "Jited and normal", "Jited and optimized", "sparse matrix"], loc="upper left", bbox_to_anchor=(0.075, .995))

# Graphs for the second plot
ax2.plot(dimensions, normal_dim_data, marker="s")
ax2.plot(dimensions, opt_dim_data, marker="o")
ax2.plot(dimensions, jited_normal_dim_data, marker="d")
ax2.plot(dimensions, jited_opt_dim_data, marker="*")
ax2.plot(dimensions, sparse_dim_data, marker=".")

# Labels and legends for th second plot
ax2.set_xlabel("matrix dimensions")
ax2.set_yscale("log")
ax2.set_title(f"Fixed number of calls: {calls}")

# Show everything
ax1.grid(True)
ax2.grid(True)
plt.tight_layout()
plt.draw()
plt.show()
