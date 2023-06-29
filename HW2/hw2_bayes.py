import os
import codecs
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch as T

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

file_list = [
    "t10k-images.idx3-ubyte",
    "t10k-labels.idx1-ubyte",
    "train-images.idx3-ubyte",
    "train-labels.idx1-ubyte",
]
    
def b2i(b):
    return int(codecs.encode(b, 'hex'), 16)


def discrete_classifier(x_train, y_train, x_test, y_test):
    ### get prior
    prior = get_prior(y_train)
    
    ### get likelihood
    likelihood = get_likelihood(x_train, y_train)
    
    
    wrong_predict = 0
    posteriors = []
    ### compute the posterior
    for i in range(len(x_test)):
        posterior = np.log(prior)
        
        for l in range(10):
            for pixel in range(len(x_test[0])):
                posterior[l] += np.log(likelihood[l, pixel, x_test[i][pixel] // 8])
                
        posterior /= np.sum(posterior)
        posteriors.append(posterior)
        
        ### Maximum A Posterior -> find min because posterior is positive
        predict = np.argmin(posterior)
        if predict != y_test[i] :
            wrong_predict += 1
    
    return posteriors, likelihood, float(wrong_predict) / len(x_test)


def continuous_classifier(x_train, y_train, x_test, y_test):
    ### using the mean and variance of the Gaussian distribution to compute posterior
    
    ### get prior
    prior = get_prior(y_train)
    
    ### get mean and variance
    mean, variance = MLE_Gaussian(x_train, y_train, prior)
    
    ## compute the posterior
    wrong_predict = 0
    posteriors = []
    
    for i in range(len(x_test)):
        posterior = np.log(prior)
        
        for l in range(10):
            for pixel in range(len(x_test[0])):
                if variance[l, pixel] == 0:
                    continue
                posterior[l] -= np.log(variance[l, pixel]) / 2.0
                posterior[l] -= np.square(x_test[i, pixel] - mean[l, pixel]) / variance[l, pixel]
                
        posterior /= np.sum(posterior)
        posteriors.append(posterior)
        
        ### Maximum A Posterior -> find min because posterior is positive
        predict = np.argmin(posterior)
        if predict != y_test[i] :
            wrong_predict += 1
    
    return posteriors, mean, float(wrong_predict) / len(x_test)
    
    
def get_prior(label):
    prior = np.zeros(10, dtype=float)    # labels have 10 type
#     prior = T.tensor(prior).to(device)
    
    for i in range(len(label)):
        prior[label[i]] += 1
        
    return prior / len(label)


def get_likelihood(data, label):
    likelihood = np.zeros((10, len(data[0]), 32), dtype=float)
#     likelihood = T.tensor(likelihood).to(device)
    
    for i in range(len(data)):
        for pixel in range(len(data[0])):
            likelihood[label[i], pixel, data[i][pixel] // 8] += 1
            
    ### get frequency
    total_num = np.sum(likelihood, axis=2)
#     total_num = T.tensor(total_num).to(device)

    for l in range(10):
        for pixel in range(len(data[0])):
            likelihood[l, pixel, :] /= total_num[l, pixel]
    
    ### pseudo count
    likelihood[likelihood == 0] = 0.00001
    
    return likelihood
    
def MLE_Gaussian(data, label, weight):
    ### compute the mean and variance of each pixel in each class
    
    label_num = weight * len(data)
    
    ### mean
    mean = np.zeros((10, len(data[0])), dtype=float)
    
    for i in range(len(data)):
        mean[label[i], :] += data[i, :]
    for l in range(10):
        mean[l, :] /= label_num[l]
    
    
    ### variance
    variance = np.zeros((10, len(data[0])), dtype=float)
    
    for i in range(len(data)):
        variance[label[i], :] += np.square(data[i, :] - mean[label[i], :])
    for l in range(10):
        variance[l, :] /= label_num[l]
    
        
    return mean, variance


def parse_arguments():
    parser = argparse.ArgumentParser(description='HW2 Naive Bayes Classifier')
    parser.add_argument('--mode', default=0, help='discrete=0, continuous=1', type=int)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    mode = args.mode
    
    ### load data
    
    #### load x_train, y_train, x_test, y_test 
    #### and record train_col, train_row, test_col, test_row
    
    ### load x_train
    with open("." + os.sep + "train-images.idx3-ubyte", 'rb') as f:
        data = f.read()
        type = b2i(data[0:4])
        dataLen = b2i(data[4:8])
        train_row = b2i(data[8:12])
        train_col = b2i(data[12:16])
        
        x_train = np.frombuffer(data, dtype=np.uint8, offset=16)    ### data start from pos=16
        x_train = x_train.reshape(dataLen, train_row * train_col)
#         x_train = T.tensor(x_train).to(device)
        
    ### load y_train
    with open("." + os.sep + "train-labels.idx1-ubyte", 'rb') as f:
        data = f.read()
        type = b2i(data[0:4])
        dataLen = b2i(data[4:8])

        y_train = np.frombuffer(data, dtype=np.uint8, offset=8)
        y_train = y_train.reshape(dataLen)
#         y_train = T.tensor(y_train).to(device)
    
    ### load x_test
    with open("." + os.sep + "t10k-images.idx3-ubyte", 'rb') as f:
        data = f.read()
        type = b2i(data[0:4])
        dataLen = b2i(data[4:8])
        test_row = b2i(data[8:12])
        test_col = b2i(data[12:16])

        x_test = np.frombuffer(data, dtype=np.uint8, offset=16)    ### data start from pos=16
        x_test = x_test.reshape(dataLen, test_row * test_col)
#         x_test = T.tensor(x_test).to(device)
        
    ### load y_test
    with open("." + os.sep + "t10k-labels.idx1-ubyte", 'rb') as f:
        data = f.read()
        type = b2i(data[0:4])
        dataLen = b2i(data[4:8])

        y_test = np.frombuffer(data, dtype=np.uint8, offset=8)
        y_test = y_test.reshape(dataLen)
#         y_test = T.tensor(y_test).to(device)
        
        
    #### so far we have: x_train, y_train, x_test, y_test, train_col, train_row, test_col, test_row
    if mode == 0:
        ### discrete mode
        posteriors, likelihood, wrong_rate = discrete_classifier(x_train, y_train, x_test, y_test)
        
        ### displsay results
        
        ### print posteriors
        for i in range(len(posteriors)):
            print('Posterior (in log scale):')
            for l in range(10):
                print(f'{l}: {posteriors[i][l]}')
            print(f'Prediction: {np.argmin(posteriors[i])}, Ans: {y_test[i]}\n')
            
        ### print graph
        ones = np.sum(likelihood[:, :, 16:32], axis=2)
        zeros = np.sum(likelihood[:, :, 0:16], axis=2)
        graph = (ones >= zeros)
        
        for l in range(10):
            print(f'{l}:')
            for r in range(test_row):
                for c in range(test_col):
                    print('1', end=' ') if graph[l, r * test_col + c] else print('0', end=' ')
                print('')
            print('')
    else:
        posteriors, means, wrong_rate = continuous_classifier(x_train, y_train, x_test, y_test)
        
    ### display results
    ### print posteriors
        for i in range(len(posteriors)):
            print('Posterior (in log scale):')
            for l in range(10):
                print(f'{l}: {posteriors[i][l]}')
            print(f'Prediction: {np.argmin(posteriors[i])}, Ans: {y_test[i]}\n')
            
            
        ### print graph
        graph = (means >= 128)
        ##################################################
        for l in range(10):
            print(f'{l}:')
            for r in range(test_row):
                for c in range(test_col):
                    print('1', end=' ') if graph[l, r * test_col + c] else print('0', end=' ')
                print('')
            print('')
    print(f'Error rate: {wrong_rate}')
    