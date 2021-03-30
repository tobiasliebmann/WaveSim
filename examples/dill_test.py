import dill as dl
import numpy as np


def func(x):
    return 2 * x


test_data = [np.array([1, 2, 3]), func]

with open("test_file.txt", "wb") as file:
    string = dl.dumps(test_data)
    print(string)
    file.write(string)

with open("test_file.txt", "rb") as file:
    print(dl.loads(file.read()))
