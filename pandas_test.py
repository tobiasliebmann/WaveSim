import pandas as pd

import numpy as np


# df = pd.read_csv("test_data2.csv")
new_row = pd.DataFrame({"First": "Test 1", "Second": "Test 2", "Third": [np.array([[1., 2.], [3., 4.], [5., 6.]]).tolist()]})
with open("test_data2.csv", "a") as file:
    new_row.to_csv(file, index=False, sep=";")
