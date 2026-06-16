import pandas as pd
import matplotlib.pyplot as plt

# read CSV, skip the first two header rows
df = pd.read_csv("Fig_7c_data.csv", skiprows=2, header=None)

# assign column names manually
df.columns = [
    "SSTn_late_X", "SSTn_late_Y",
    "SSTn_early_X", "SSTn_early_Y",
    "SSTp_late_X", "SSTp_late_Y",
    "SSTp_early_X", "SSTp_early_Y"
]

plt.figure(figsize=(5,4))

# SSTn (grey)
plt.plot(df["SSTn_late_X"], df["SSTn_late_Y"], color="grey", linestyle="-", label="SSTn late")
plt.plot(df["SSTn_early_X"], df["SSTn_early_Y"], color="grey", linestyle="--", label="SSTn early")

# SSTp (black)
plt.plot(df["SSTp_late_X"], df["SSTp_late_Y"], color="black", linestyle="-", label="SSTp late")
plt.plot(df["SSTp_early_X"], df["SSTp_early_Y"], color="black", linestyle="--", label="SSTp early")

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()
plt.show()