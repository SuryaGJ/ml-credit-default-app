import pandas as pd

df = pd.read_csv("UCI_Credit_Card.csv")
df.sample(300).to_csv("sample_test.csv", index=False)
