import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def data_to_matrix(data, basis_num):
    ### build A
    n = basis_num
    A = np.zeros([len(df), n])
    
    for i, x in enumerate(df[0]):
        for j in range(n):
            A[i][j] = pow(x, j)
    
    
    return A

def LU_decomposition(ATA, basis_num):
    n = basis_num
    ### LU decomposition
    L = np.zeros([n, n])
    U = np.zeros([n, n])
    
    for i in range(n):
        L[i][i] = 1
        
        if i == 0:
            U[0][0] = ATA[0][0]
            for j in range (1, n):
                U[0][j] = ATA[0][j]
                L[j][0] = ATA[j][0] / U[0][0]
        else:
            for j in range(i, n):
                temp = 0
                for k in range(0, i):
                    temp = temp + L[i][k] * U[k][j]
                U[i][j] = ATA[i][j] - temp
                
            for j in range(i+1, n):
                temp = 0
                for k in range(0, i):
                    temp = temp + L[j][k] * U[k][i]
                L[j][i] = (ATA[j][i] - temp) / U[i][i]
            
    return L, U

def triangular_inverse(A, n):
    ### the implementation of lower triangular matrix
    if n == 2:
        temp = 1/(A[0][0]*A[1][1])
        inv_A = np.array([(A[1][1]*temp, 0), (-A[1][0]*temp, A[0][0]*temp)])
        return inv_A
    
    inv_A = np.eye(n)
    
    for i in range(n):
        for j in range (i+1): 
            inv_A[i][j] /= A[i][i]
        
        for j in range(i+1, n):
            temp = A[j][i]
            
            for k in range(j):
                inv_A[j][k] -= temp*inv_A[i][k]
    
    return inv_A

def LSE(A, basis_num, Lambda=0):
    ### solve Ax = b
    n = basis_num
    ATA = np.matmul(A.T, A)
    
    ### A + lambda*I
    if Lambda != 0:
        ATA += Lambda * np.eye(n)
        
    L, U = LU_decomposition(ATA, n)
    inv_L = triangular_inverse(L, n)
    inv_U = triangular_inverse(U.T, n).T
    
#     print('inv_U:\n', inv_U)
#     print('correct inv_U:\n', np.linalg.inv(U))
    
    x = np.matmul(A.T, df[1])
    x = np.matmul(inv_L, x)
    x = np.matmul(inv_U, x)
    
    return x

def newton_method(A, basis_num):
    n = basis_num
    
    b = df[1].values.reshape(-1, 1)
    ### solve Ax = b
    
    ATA = np.matmul(A.T, A)
    
    ### random initialize
    x = (np.random.rand(n, 1) * 2 - [[0.5] for _ in range(n)])
    
    ### jaccobian matrix
    gradient = (np.matmul(ATA, x) - np.matmul(A.T, b))
    
    ### hessian matrix
    hessian = ATA
    ### before find inverse, LU decomposition
    L, U = LU_decomposition(hessian, n)
    inv_L = triangular_inverse(L, n)
    inv_U = triangular_inverse(U.T, n).T
    inv_hessian = np.matmul(inv_U, inv_L)
    
    x = x - np.matmul(inv_hessian, gradient)
    
    return x.reshape(-1)

def compute_error(A, x, y):
    ### use RSE
    error_vec = np.matmul(A, x) - y
    error_vec = list(map(lambda x: x**2, error_vec))
    return sum(error_vec)

if __name__ == '__main__':
    n = int(input('using dim: '))
    Lambda = int(input('using lambda:'))
    print()
    df = pd.read_csv('testfile.txt', sep=',', header=None)
    A = data_to_matrix(df, n)
    LSE_coef = LSE(A, n, Lambda)
#     print(LSE_coef)
    LSE_total_error = compute_error(A, LSE_coef, df[1])
#     print('total_error: ', total_error)
#     print()
    newton_coef = newton_method(A, n)
#     print(coef)
    newton_total_error = compute_error(A, newton_coef, df[1])
#     print('total_error: ', total_error)
    
    ### output
    print('LSE:')
    LSEoutput = 'Fitting line: '
    
    for degree in range(n-1, -1, -1):
        if degree !=  n-1 :
            if LSE_coef[degree] >= 0:
                LSEoutput += ' + '
            else:
                LSEoutput += ' '
        
        if degree != 0:
            LSEoutput += f'{LSE_coef[degree]: .11f}X^{degree}'
        else:
            LSEoutput += f'{LSE_coef[degree]: .11f}'
    
    print(LSEoutput)
    print(f'Total error: {LSE_total_error: .11f}')
    
    print()
    
    print("Newton's Method:")
    newton_output = 'Fitting line: '
    
    for degree in range(n-1, -1, -1):
        if degree !=  n-1 :
            if newton_coef[degree] >= 0:
                newton_output += ' +'
            else:
                newton_output += ' '
        
        if degree != 0:
            newton_output += f'{newton_coef[degree]: .11f}X^{degree}'
        else:
            newton_output += f'{newton_coef[degree]: .11f}'
    
    print(newton_output)
    print(f'Total error: {newton_total_error: .11f}')
    
    
    ### plot 
    
    ### the function of LSE and newtion's method
    LSE_func = 0
    newton_func = 0
    for degree in range(n-1, -1, -1):
        LSE_func += LSE_coef[degree] * (df[0].values ** degree)
        newton_func += newton_coef[degree] * (df[0].values ** degree) 
    plt.figure(1)
    
    plt.subplot(211)
    plt.scatter(df[0].values, df[1].values, c='r')
    plt.plot(df[0].values, LSE_func)
    
    plt.subplot(212)
    plt.scatter(df[0].values, df[1].values, c='r')
    plt.plot(df[0].values, newton_func)
    plt.show()