from poly_data_generator import poly_linear_data_generator
import numpy as np
import matplotlib.pyplot as plt
import argparse
from copy import deepcopy

def baysian_linear_regression(b, n, a, w):
    ### b: precision for initial prior
    ### n, a, w: required for the polynomial basis linear model generator
    ### 
    
    ### set initial values 
    prior_mean = np.zeros(n)
    precision = b
    inv_var = 1.0 / a
    epsilon = 0.0001
    count = 1
    
    ### plot info
    points = []
    mean_10 = 0
    cov_10 = 0
    mean_50 = 0
    var_50 = 0
    
    while True:
        x, y = poly_linear_data_generator(n, a, w)
        points.append([x, y])
        
        Phi = designed_matrix(x, n)
        if count == 1:
            posterior_cov = np.linalg.inv(precision * np.identity(n) +  inv_var * np.matmul(Phi.T, Phi))
            posterior_mean = inv_var * np.matmul(posterior_cov, Phi.T) * y
            count += 1
        else:
            posterior_cov = np.linalg.inv(inv_var * np.matmul(Phi.T, Phi) + np.linalg.inv(prior_cov))
            posterior_mean = np.matmul(posterior_cov, np.matmul(np.linalg.inv(prior_cov), prior_mean) + inv_var * Phi.T * y)
            count += 1
            
        predictive_mean = np.matmul(Phi, posterior_mean)
        predictive_var = a + np.matmul(np.matmul(Phi, posterior_cov), Phi.T)
        
        ### show the result
        print(f'Add data point ({x:.5f}, {y:.5f}):\n')
        print(f'Posterior mean:')
        for i in range(n):
            print(f'{posterior_mean[i][0]:.10f}')
            
        print(f'\nPosterior variance:')
        for i in range(n):
            for j in range(n):
                if j != n - 1:
                    print(f'{posterior_cov[i, j]:.10f},', end='\t')
                else:
                    print(f'{posterior_cov[i, j]:.10f}')
        
        print(f'\nPredictive distribution ~ N({predictive_mean[0][0]:.5f}, {predictive_var[0][0]:.5f})\n')
        
        ### plot info
        if count == 10:
            mean_10 = deepcopy(posterior_mean)
            cov_10 = deepcopy(posterior_cov)
        if count == 50:
            mean_50 = deepcopy(posterior_mean)
            cov_50 = deepcopy(posterior_cov)
        
        ### check convergence
        if np.linalg.norm(prior_mean - posterior_mean) <= epsilon and count >= 50:
            break
        
        ### update
        prior_mean = posterior_mean
        prior_cov = posterior_cov

    return points, posterior_mean, posterior_cov, mean_10, cov_10, mean_50, cov_50

def designed_matrix(x, n):
    Phi = np.zeros((1, n))
    for i in range(n):
        Phi[0, i] = np.power(x, i)
        
    return Phi

def parse_arguments():
    parser = argparse.ArgumentParser(description='HW3 baysian linear regression')
    parser.add_argument('--precision', default=1, type=float)
    parser.add_argument('--variance', default=1, type=float)
    parser.add_argument('--weight', nargs='+', help='e.g. w 1 2 3 means w = [1, 2, 3]', default=[1,2,3,4], type=float)
    
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_arguments()
    b = args.precision
    a = args.variance
    w = args.weight
    n = len(w)
    
    points, posterior_mean, posterior_cov, mean_10, cov_10, mean_50, cov_50 = baysian_linear_regression(b, n, a, w)
    
    ### show the result
    x = np.linspace(-2.0, 2.0, 100)
    points = np.transpose(points)
    
    #### Ground Truth
    plt.subplot(221)
    plt.title('Ground Truth')
    f = np.poly1d(np.flip(w))
    y = f(x)
    plt.plot(x, y, color='k')
    plt.plot(x, y + a, color='r')
    plt.plot(x, y - a, color='r')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-15.0, 25.0)
    
    #### Predict Result
    plt.subplot(222)
    plt.title('Predict Result')
    f = np.poly1d(np.flip(np.reshape(posterior_mean, n)))
    y = f(x)
    var = np.zeros(100)
    for i in range(100):
        Phi = designed_matrix(x[i], n)
        var[i] = a + np.matmul(Phi, np.matmul(posterior_cov, Phi.T))

    plt.scatter(points[0], points[1], s=1)
    plt.plot(x, y, color='k')
    plt.plot(x, y + var, color='r')
    plt.plot(x, y - var, color='r')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-15.0, 25.0)
    
    #### After 10 times
    plt.subplot(223)
    plt.title('After 10 times')
    f = np.poly1d(np.flip(np.reshape(mean_10, n)))
    y = f(x)
    var = np.zeros(100)
    for i in range(100):
        Phi = designed_matrix(x[i], n)
        var[i] = a + np.matmul(Phi, np.matmul(cov_10, Phi.T))

    plt.scatter(points[0][:10], points[1][:10], s=1)
    plt.plot(x, y, color='k')
    plt.plot(x, y + var, color='r')
    plt.plot(x, y - var, color='r')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-15.0, 25.0)
    
    #### After 50 times
    plt.subplot(224)
    plt.title('After 50 times')
    f = np.poly1d(np.flip(np.reshape(mean_50, n)))
    y = f(x)
    var = np.zeros(100)
    for i in range(100):
        Phi = designed_matrix(x[i], n)
        var[i] = a + np.matmul(Phi, np.matmul(cov_50, Phi.T))

    plt.scatter(points[0][:50], points[1][:50], s=1)
    plt.plot(x, y, color='k')
    plt.plot(x, y + var, color='r')
    plt.plot(x, y - var, color='r')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-15.0, 25.0)
    
    plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
    plt.show()