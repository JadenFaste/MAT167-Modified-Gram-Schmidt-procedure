import numpy as np

def modified_gram_schmidt(A):
    n, m = A.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))
    
    for k in range(m):
        R[k, k] = np.linalg.norm(A[:, k])
        Q[:, k] = A[:, k] / R[k, k]
        
        for j in range(k+1, m):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] = A[:, j] - R[k, j] * Q[:, k]
    
    return Q, R

# Given vectors
a1 = np.array([1, 0, 0])
a2 = np.array([1, 1, 0])
a3 = np.array([1, 1, 1])

# Form the matrix A
A = np.column_stack((a1, a2, a3))

# Perform the modified Gram-Schmidt process
Q, R = modified_gram_schmidt(A)

print("Q matrix:")
print(Q)
print("\nR matrix:")
print(R)

# Verify the result
A_prime = np.dot(Q, R)
print("\nA' (Q * R):")
print(A_prime)

# Compute the relative error
relative_error = np.linalg.norm(A - A_prime) / np.linalg.norm(A)
print("\nRelative Error:")
print(relative_error)
