import pandas as pd

df = pd.read_csv("/home/tobias/Schreibtisch/test_data.csv", delimiter=";")
print(df)
new_row = pd.DataFrame([["Sebastian", "Liebmann", [1, 2, 3, 4, 5]]])
with open("/home/tobias/Schreibtisch/test_data.csv", "a") as file:
    new_row.to_csv(file, header=False, index=False, sep=";")


df2 = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
print(df2)
df3 = pd.DataFrame([[7, 8]], columns=["A", "B"])
print(df2.append(df3))

