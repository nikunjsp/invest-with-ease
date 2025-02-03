import pandas as pd
import numpy

df = pd.read_csv("2final.csv")
df.reset_index(inplace=True, drop=True)
df.to_csv("2final.csv")
