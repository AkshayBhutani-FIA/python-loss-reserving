import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../data/claims_triangle.csv")
df.set_index("AccidentYear", inplace=True)

cum = df.cumsum(axis=1)

link_ratios = cum.iloc[:, 1:].values / cum.iloc[:, :-1].values
factors = np.nanmean(link_ratios, axis=0)

proj = cum.copy()
for i in range(len(factors)):
    proj.iloc[:, i + 1] = proj.iloc[:, i] * factors[i]

ultimate = proj.iloc[:, -1]
latest = cum.apply(lambda row: row.dropna().iloc[-1], axis=1)
ibnr = ultimate - latest

summary = pd.DataFrame({
    "Latest": latest,
    "Ultimate": ultimate,
    "IBNR": ibnr
})

print(summary)

summary[["Latest", "Ultimate"]].plot(kind="bar")
plt.title("Latest vs Ultimate Losses")
plt.tight_layout()
plt.show()
