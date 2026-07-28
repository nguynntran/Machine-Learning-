import numpy as np
import pandas as pd

np.random.seed(42)

# number of points
n = 500

# generate random features
feature1 = np.random.uniform(-2, 8, n)
feature2 = np.random.uniform(-2, 8, n)

# linear boundary equation:
# label = 1 if feature1 + feature2 > 6
labels = (feature1 + feature2 > 6).astype(int)

# create dataframe
data = pd.DataFrame({
    "feature1": feature1,
    "feature2": feature2,
    "label": labels
})

# save to csv
data.to_csv("data_large.csv", index=False)

print("Dataset created: data_large.csv")
print(data.head())