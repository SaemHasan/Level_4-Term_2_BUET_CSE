import numpy as np

n = int(input("Enter n : \n"))
# Generate a random n*n invertible matrix
A = np.random.randint(0,n*n,size=(n,n))
det = np.linalg.det(A)
while det == 0:
    A = np.random.randint(0,n*n,size=(n,n))
    det = np.linalg.det(A)

# symmetric matrix
A = A + A.T

print("A = \n",A)

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues : \n",eigenvalues)
print("Eigenvectors : \n",eigenvectors)

# reconstruct matrix
# B = eigenvectors.dot(np.diag(eigenvalues)).dot(np.linalg.inv(eigenvectors))
B = eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T)
print("Reconstructed matrix : \n",B)

# check the reconstruction
print("Is the reconstructed matrix equal to the original matrix? : ", np.allclose(A, B))

