from gaussion_data_generator import univariate_gaussian_data_generator
import numpy as np
import argparse

def poly_linear_data_generator(n, a, w):
    ### generate x ~ Uniform(-1, 1)
    x = np.random.uniform(-1, 1)
    
    ### bias e ~ N(0, a)
    e = np.random.normal(0, a)
    
    y = 0
    for i in range(n):
        y += np.power(x, i) * w[i]
        
    return x, y + e



def parse_arguments():
    parser = argparse.ArgumentParser(description='HW3 polynomial data generator')
#     parser.add_argument('--basis', default=1.0, type=float)
    parser.add_argument('--variance', default=0, type=float)
    parser.add_argument('--weight', nargs='+', help='e.g. w 1 2 3 means w = [1, 2, 3]', default=[1,2,3,4], type=float)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    variance = args.variance
    w = args.weight
    n = len(w)
    x, y = poly_linear_data_generator(n, variance, w)
    print(f'generate data ({x:6.5f}, {y:6.5f})')