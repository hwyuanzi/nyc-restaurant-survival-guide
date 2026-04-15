import numpy as np
from sklearn.preprocessing import StandardScaler

x = np.array([0, 1, 0, 1, 1], dtype=np.float32).reshape(-1, 1)
x_weighted = x * 2.8

scaler = StandardScaler()
x_scaled1 = scaler.fit_transform(x)
x_scaled2 = scaler.fit_transform(x_weighted)

print("Original scaled:", x_scaled1.flatten())
print("Weighted scaled:", x_scaled2.flatten())
