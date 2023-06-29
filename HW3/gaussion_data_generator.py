import numpy as np
import argparse
### uniform distribution -> standard Gaussian distribution -> general Gaussion distribution
def univariate_gaussian_data_generator(m, s):
    ### randam choose data point from unifrom distribution
    U1 = np.random.uniform(0, 1)
    U2 = np.random.uniform(0, 1)
    
    ### the constant R, theta
    R = np.sqrt(-2 * np.log(U1))
    theta = 2 * np.pi * U2
    ### standard Gaussion distribution
    Z0 = R * np.cos(theta)
    Z1 = R * np.sin(theta)
    
    x = m + np.sqrt(s) * Z0
    y = m + np.sqrt(s) * Z1
    return x, y


def parse_arguments():
    parser = argparse.ArgumentParser(description='HW3 gaussian data generator')
    parser.add_argument('--variance', default=1.0, type=float)
    parser.add_argument('--mean', default=0, type=float)
    
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_arguments()
    mean = args.mean
    variance = args.variance
    
    x, y = univariate_gaussian_data_generator(mean, variance)
    
    print(f'generate data ({x:10.5f}, {y:10.5f}) ~ N({mean:.2f}, {variance:.2f})')