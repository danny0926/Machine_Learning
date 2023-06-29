from gaussion_data_generator import univariate_gaussian_data_generator
import numpy as np
import argparse

### sequential estimator
def sequential_estimator(m, s):
    ### simple constrain: # of loops
    print(f'Data point source function: N({m}, {s})\n')
    N = 0
    current_sum = 0
    current_sum2 = 0
    last_mean = 0
    last_var = 0
    epsilon = 0.001
    
    while True:
        x, _ = univariate_gaussian_data_generator(m, s)
        
        N += 1
        current_sum += x
        current_sum2 += x ** 2
        
        mean = current_sum / N
        var = (current_sum2 - (current_sum ** 2) / N) / (N - 1) if N > 1 else 0
        
        print(f'Add data point: {x}')
        print(f'Mean = {mean:.16f}\tVariance = {var:.16f}')
        
        if np.linalg.norm(last_mean - mean) <= epsilon:
            break
        
        last_mean = mean
        last_var = var
        
def parse_arguments():
    parser = argparse.ArgumentParser(description='HW3 sequential estimator')
    parser.add_argument('--variance', default=1.0, type=float)
    parser.add_argument('--mean', default=0, type=float)
    
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_arguments()
    mean = args.mean
    variance = args.variance
    
    sequential_estimator(mean, variance)     