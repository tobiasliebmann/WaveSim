from scipy import sparse as sp
import numpy as np
import time as tm
import numba as nb


def create_matrix(matrix_dimension: int) -> np.ndarray:
    """
    This function creates a quadratic matrix with a populated diagonal, populated off-diagonals and dimension
    matrix_dimension. All entries in the diagonal are (1 - 2 * number) while the entreis in the off-diagonal are the
    number number.
    :param matrix_dimension: Dimension of the quadratic matrix which is returned.
    :return: The matrix constructed as described above.
    """
    number = np.random.random()
    matrix = (1 - 2 * number) * np.identity(matrix_dimension)
    np.fill_diagonal(matrix[1:], number)
    np.fill_diagonal(matrix[:, 1:], number)
    return matrix


@nb.jit(nopython=True)
def jit_mul(matrix1, matrix2):
    return np.dot(matrix1, matrix2)


dim = 2000

state_matrix = np.random.rand(dim, dim)
time_step_matrix = create_matrix(dim)
time_step_matrix_sparse = sp.csr_matrix(time_step_matrix)

start_normal = tm.time()
np.dot(time_step_matrix, state_matrix)
end_normal = tm.time()

start_sparse = tm.time()
time_step_matrix_sparse.dot(state_matrix)
end_sparse = tm.time()

first_start_jit = tm.time()
jit_mul(time_step_matrix, state_matrix)
first_end_jit = tm.time()

second_start_jit = tm.time()
jit_mul(time_step_matrix, state_matrix)
second_end_jit = tm.time()

print(f"The numpy matrix multiplication took {end_normal - start_normal} s.")
print(f"The sparse matrix multiplication took {end_sparse - start_sparse} s.")
print(f"The jited sparse matrix multiplication took {first_end_jit - first_start_jit} s on the first call.")
print(f"The jited sparse matrix multiplication took {second_end_jit - second_start_jit} s on the second call.")
