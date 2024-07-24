import numpy as np

residual_matrix = np.array([
    [0.76, 2.23, np.nan, 4.1],
    [0.83, 4.04, np.nan, 3.79],
    [np.nan, 2.69, 3.93, 3.61],
    [3.05, 1.56, 2.76, np.nan]
])

# nanを0に置換
residual_matrix = np.nan_to_num(residual_matrix)

# 要素ごとに2乗し、その合計を求める
frobenius_norm_squared = np.sum(residual_matrix**2)

print(frobenius_norm_squared)
