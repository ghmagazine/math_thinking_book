import numpy as np

# 残差行列を定義
residual_matrix = np.array([
    [0.99, 2.54, np.nan, 4.47],
    [1.06, 4.36, np.nan, 4.18],
    [np.nan, 2.85, 3.97, 3.82],
    [3.24, 1.77, 2.84, np.nan]
])

# nanを0に置換
residual_matrix = np.nan_to_num(residual_matrix)

# 要素ごとに2乗し、その合計を求める
frobenius_norm_squared = np.sum(residual_matrix**2)

print(frobenius_norm_squared)
