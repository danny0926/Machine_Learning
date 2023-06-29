import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize


def gaussian_process(x_train, y_train, noise, alpha=1.0, length_scale=1.0, name=''):
    
    point_num = 1000
    ### get test data
    x_test = np.linspace(-60, 60, point_num).reshape(-1, 1)
    
    ### get covariance matrix
    cov_matrix = rational_quadratic_kernel(x_train, x_train, alpha, length_scale)
    
    ### get kernel of test data to test data
    kernel_test = np.add(rational_quadratic_kernel(x_test, x_test, alpha, length_scale), np.eye(len(x_test)) / noise)
    
    ### get kernel of test data to train data
    kernel_train_test = rational_quadratic_kernel(x_train, x_test, alpha, length_scale)
    
    ### get mean and variance
    mean = kernel_train_test.T.dot(np.linalg.inv(cov_matrix)).dot(y_train).ravel()
    variance = kernel_test - kernel_train_test.T.dot(np.linalg.inv(cov_matrix)).dot(kernel_train_test)
    
    ### get 95% confidence interval
    upper_bound = mean + 1.96 * variance.diagonal()
    lower_bound = mean - 1.96 * variance.diagonal()
    
    ### plot
    plt.xlim([-60, 60])
    plt.title(f"{name} Gaussian Process")
    plt.scatter(x_train, y_train, color='black')
    plt.plot(x_test.ravel(), mean, color='blue')
    plt.fill_between(x_test.ravel(), upper_bound, lower_bound, color='cadetblue', alpha=0.5)
    plt.savefig('GaussianProcess_' + name + '.png')
    plt.show()
    
    
def rational_quadratic_kernel(x, y, alpha, length_scale):
    return 1.0 * np.power(1 + cdist(x, y, 'sqeuclidean') / (2 * alpha * length_scale * length_scale), -alpha)


def marginal_log_likelihood(theta):
    
    global x_train, y_train
    point_num = len(x_train)
    cov_matrix = rational_quadratic_kernel(x_train, x_train, theta[0], theta[1])
    
    return 0.5 * np.log(np.linalg.det(cov_matrix)) + 0.5 * y_train.ravel().T.dot(np.linalg.inv(cov_matrix)).dot(y_train.ravel()) + point_num / 2.0 * np.log(2.0 * np.pi)

    
if __name__ == '__main__':
    ### load data
    data = np.loadtxt('data/input.data')
    
    x_train = data[:, 0].reshape(-1, 1)
    y_train = data[:, 1].reshape(-1, 1)
    points = len(x_train)
    gaussian_process(x_train, y_train, noise=5.0, name='Original')
    
    ### optimized parameters
    theta = np.array([1.0, 1.0])
    res = minimize(marginal_log_likelihood, theta)
    
    #print(res)
    alpha, length_scale = res.x
    print(alpha, length_scale)
    gaussian_process(x_train, y_train, noise=5.0, alpha=alpha, length_scale=length_scale, name='Optimize')