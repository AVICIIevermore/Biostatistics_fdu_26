import numpy as np
d = np.load('D:/Fudan/GitHub repositories/MMMD-boost-kernel-two-sample/PathMNIST/pathmnist.npz')
y = np.concatenate([d['train_labels'], d['val_labels'], d['test_labels']]).ravel()
for c in [3, 5, 6, 8]:
    print('class', c, ':', (y == c).sum(), 'samples')
