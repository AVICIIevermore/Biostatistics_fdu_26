import numpy as np

path = "pathmnist.npz"
data = np.load(path)

print(data.files)
for k in data.files:
    arr = data[k]
    print(k, arr.shape, arr.dtype, arr.min(), arr.max())