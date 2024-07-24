import numpy as np

def matrix_factorization(R, P, Q, steps, alpha):
    for step in range(steps):
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if not np.isnan(R[i, j]):
                    error = R[i, j] - np.dot(P[i, :], Q[:, j])
                    P[i, :] += 2 * alpha * error * Q[:, j]
                    Q[:, j] += 2 * alpha * error * P[i, :]
        PQ = np.dot(P, Q)
        PQ = np.clip(PQ, 0, 5)
        if np.all(np.isnan(R) | (R == PQ)):
            break
    return P, Q, PQ

R = np.array([
    [2, 3, np.nan, 5],
    [2, 5, np.nan, 5],
    [np.nan, 3, 4, 4],
    [4, 2, 3, np.nan]
])

np.random.seed(42)
P = np.random.rand(4, 2)
Q = np.random.rand(2, 4)
alpha = 0.01
steps = 1

print("Initial P:")
print(P)
print("Initial Q:")
print(Q)

P, Q, PQ = matrix_factorization(R, P, Q, steps, alpha)

print("Final P:")
print(P)
print("Final Q:")
print(Q)
print("PQ:")
print(PQ)
