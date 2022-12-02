import numpy as np

n = int(input("Enter n : \n"))
m = int(input("Enter m : \n"))

# Generate a random n*m matrix
A = np.random.randint(0,n*m,size=(n,m))

print("A = \n",A)

U, s, V = np.linalg.svd(A, full_matrices=False)

print("U = \n",U)
print("s = \n",s)
print("V = \n",V)

# calculate the pseudoinverse
# Calculate the Moore-Penrose Pseudoinverse using NumPy’s builtin function
C = np.linalg.pinv(A)
print("Moore-Penrose Pseudoinverse using numpy : \n",C)
# Calculate the Moore-Penrose Pseudoinverse again using Eq. 2.47 in the book
D = V.T.dot(np.diag(1/s)).dot(U.T)
print("Moore-Penrose Pseudoinverse using Eq. 2.47 : \n",D)

# check two pseudoinverse
print("Is the two pseudoinverse equal to each other? : ", np.allclose(C, D))