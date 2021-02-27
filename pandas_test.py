import pandas as pd

# df = pd.read_csv("test_data2.csv")
new_row = pd.DataFrame({"First": "Test 1", "Second": "Test 2", "Third": [[1, 2, 3, 4, 5]]})
with open("test_data2.csv", "a") as file:
    new_row.to_csv(file, header=True, index=False, sep=";")
