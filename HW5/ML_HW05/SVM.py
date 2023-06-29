import numpy as np
import argparse
from scipy.spatial.distance import cdist
from libsvm.svmutil import *
import time
import tqdm

def linear_poly_rbf_comparison(x_train, y_train, x_test, y_test):
    ### compare linear, polynomial and rbf kernel
    kernels = ['Linear', 'Polynomial', 'RBF']
    
    for idx, name in enumerate(kernels):
        param = svm_parameter(f'-t {idx} -q')
        prob = svm_problem(y_train, x_train)
        
        print(f'{name}: ')
        start = time.time()
        model = svm_train(prob, param)
        svm_predict(y_test, x_test, model)
        end = time.time()
        
        print(f'Cost time: {end - start:.3f}s\n')
    
    
    
def optimize(x_train, y_train, x_test, y_test):
    ### find the optimal parameters of each kernel method using grid search
    """
    Aadjust parameters:
    Linear: c
    Polynomial: c, degree, gamma, constant
    RBF: c, gamma
    """
    kernels = ['Linear', 'Polynomial', 'RBF']
    
    ### the range of parameters which can be adjusted
    cost = [np.power(10.0, i) for i in range(-3, 3)]
    degree = [i for i in range(3)]
    gamma = [1.0 / 784] + [np.power(10.0, i) for i in range(-3, 3)]
    constant = [i for i in range(-3, 3)]
    
    best_parameter = []
    max_accuracy = []
    """
    for idx, name in enumerate(kernels):
        best_param = ''
        best_acc = 0.0
        
        if name == 'Linear':
            for c in cost:
                parameters = f'-t {idx} -c {c} -q'
                param = svm_parameter(parameters + ' -v 3')
                prob = svm_problem(y_train, x_train)
                acc = svm_train(prob, param)
                
                if acc > best_acc:
                    best_acc = acc
                    best_param = parameters
            best_parameter.append(best_param)
            max_accuracy.append(best_acc)
            
        if name == 'Polynomial':
            for c in cost:
                for d in degree:
                    for g in gamma:
                        for C in constant:
                            parameters = f'-t {idx} -c {c}  -d {d} -g {g} -r {C} -q'
                            param = svm_parameter(parameters + ' -v 3')
                            prob = svm_problem(y_train, x_train)
                            acc = svm_train(prob, param)

                            if acc > best_acc:
                                best_acc = acc
                                best_param = parameters
            best_parameter.append(best_param)
            max_accuracy.append(best_acc)
            
        if name == "RBF":
            for c in cost:
                for g in gamma:
                    parameters = f'-t {idx} -c {c} -g {g} -q'
                    param = svm_parameter(parameters + ' -v 3')
                    prob = svm_problem(y_train, x_train)
                    acc = svm_train(prob, param)

                    if acc > best_acc:
                        best_acc = acc
                        best_param = parameters
            best_parameter.append(best_param)
            max_accuracy.append(best_acc)
    """        
    ### after find all optimal parameters, do the prediction
    best_parameter = ['-t 0 -c 0.01 -q', ' -t 1 -c 100.0  -d 2 -g 10.0 -r 1 -q', '-t 2 -c 100.0 -g 0.01 -q']   ### recorded by myself
    max_accuracy = [96.84, 98.26, 98.18]
    prob = svm_problem(y_train, x_train)
    for i, name in enumerate(kernels):
        print(f"{name}")
        print(f"Max accuracy: {max_accuracy[i]}")
        print(f"Best parameters: {best_parameter[i]}")
        
        param = svm_parameter(best_parameter[i])
        model = svm_train(prob, param)
        svm_predict(y_test, x_test, model)
    
    
def linear_rbf_conbination(x_train, y_train, x_test, y_test):
    ### parameters
    cost = [np.power(10.0, i) for i in range(-2, 3)]
    gamma = [1.0 / 784] + [np.power(10.0, i) for i in range(-3, 2)]
    rows, cols = x_train.shape
    
    ### find the best parameters
    linear = linear_kernel(x_train, x_train)
    best_param = '-t 4 -c 0.01 -q'
    best_gamma = 0.1
    max_acc = 0.0
    """
    for c in cost:
        for g in gamma:
            rbf = rbf_kernel(x_train, x_train, g)
            
            ### conbine two kernels: 'add'
            # combination = np.hstack((np.arange(1, rows + 1).reshape(-1, 1), linear + rbf))
            combination = linear @ rbf
            parameters = f'-t 4 -c {c} -q'
            param = svm_parameter(parameters + ' -v 3')
            prob = svm_problem(y_train, combination, isKernel=True)
            acc = svm_train(prob, param)
            
            if acc > max_acc:
                max_acc = acc
                best_param = parameters
                best_gamma = g
                
    ### print the best parameters and the max accuracy
    print('=='*30)
    print(f'Linear + RBF')
    print(f'\tMax accuracy: {max_acc}%')
    print(f'\tBest parameters: {best_param}, gamma: {best_gamma}\n')
    """
    
    ### train the model by these best parameters
    # linear = linear_kernel(x_train, x_train)
    rbf = rbf_kernel(x_train, x_train, best_gamma)
    combination = np.hstack((np.arange(1, rows + 1).reshape(-1, 1), linear + rbf))
    #combination = linear @ rbf
    model = svm_train(svm_problem(y_train, combination, isKernel=True),
                      svm_parameter(best_param))
    #print(model)
    ### predict
    rows, cols = x_test.shape
    linear = linear_kernel(x_test, x_test)
    rbf = rbf_kernel(x_test, x_test, best_gamma)
    combination = np.hstack((np.arange(1, rows + 1).reshape(-1, 1), linear + rbf))
    #combination = linear @ rbf
    """
    print(np.arange(1, rows + 1).reshape(-1, 1).shape)
    print(np.hstack((np.arange(1, rows + 1).reshape(-1, 1), linear + rbf)).shape)
    print(combination.shape)
    print((rbf + linear).shape)
    print('after compute the combination')
    """
    print(f"RBF + Linear")
    svm_predict(y_test, combination, model)
    
    
    
def linear_kernel(x, y):
    return x.dot(y.T)
    
    
def rbf_kernel(x, y, gamma):
    return np.exp(-gamma * cdist(x, y, 'sqeuclidean'))




if __name__ == '__main__':
    
    ### load data
    x_train = np.loadtxt('data/X_train.csv', delimiter=',')
    y_train = np.loadtxt('data/Y_train.csv', delimiter=',')
    x_test = np.loadtxt('data/X_test.csv', delimiter=',')
    y_test = np.loadtxt('data/Y_test.csv', delimiter=',')
    
    # linear_poly_rbf_comparison(x_train, y_train, x_test, y_test)
    
    
    # optimize(x_train, y_train, x_test, y_test)
    
    
    linear_rbf_conbination(x_train, y_train, x_test, y_test)