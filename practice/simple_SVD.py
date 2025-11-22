import numpy as np
import matplotlib.pyplot as plt

dic = ["I", "like", "enjoy", "deep", "learning", "NLP", "flying", "."]
X = np.array([[0, 2, 1, 0, 0, 0, 0, 0],
              [2, 0, 0, 1, 0, 1, 0, 0],
              [1, 0, 0, 0, 0, 0, 1, 0],
              [0, 1, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 1],
              [0, 1, 0, 0, 0, 0, 0, 1],
              [0, 0, 1, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 1, 1, 0]])
U, S, Vt = np.linalg.svd(X, full_matrices=False)

for i in range(len(dic)):
    # 8x2, 2x2, 2x8
    plt.scatter(U[i, 0], U[i, 1])
    plt.text(U[i, 0], U[i, 1], dic[i])
    
plt.show()