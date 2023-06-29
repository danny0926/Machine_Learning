import numpy as np
import argparse
import os
import codecs
from numba import jit

def em_algo(x_train, y_train, train_row, train_col):
    bin_images = x_train.copy()
    bin_images[bin_images < 128] = 0
    bin_images[bin_images >= 128] = 1
    bin_images = bin_images.astype(int)
    
    ### Initial lambda (probability of each class)
    ### use the uniform distribution here
    lam = np.full(10, 0.1)
    probability = np.random.uniform(0.0, 1.0, (10, len(bin_images[0])))
    for class_num in range(10):
        probability[class_num, :] /= np.sum(probability[class_num, :])
       
    responsibility = np.zeros((len(bin_images), 10))
    
    count = 0
    while True:
        count += 1
        old_probability = probability
        ### start EM algo
        responsibility = e_step(lam, probability, bin_images)
        lam, probability = m_step(responsibility, bin_images)
        
        difference = np.linalg.norm(probability - old_probability)
        ### print at each loop
        graph = (probability >= 0.5)
        print('')
        for class_i in range(10):
            print(f'class {class_i}')
            for row in range(train_row):
                for col in range(train_col):
                    print('1', end=' ') if graph[class_i, row * train_col + col] else print('0', end=' ')
                print('')
            print('')
        print(f'No. of Iteration: {count}, Difference: {difference:.12f}')
        
        if difference < 0.15 or count > 30:
            break
    # end while
    
    matching = find_matching(lam, probability, bin_images, y_train)
    result = predict(lam, probability, bin_images, y_train, matching)
    
    ### show the final result
    graph = (probability >= 0.5)
    list_of_matching = matching.tolist()
    print('')
    for class_i in range(10):
        result_class = list_of_matching.index(class_i)
        print(f'labeled class {class_i}')
        for row in range(train_row):
            for col in range(train_col):
                print('1', end=' ') if graph[class_i, row * train_col + col] else print('0', end=' ')
            print('')
        print('')
    
    
    ### show the confusion matrix
    error = len(bin_images)
    for class_i in range(10):
        tp, fp, tn, fn = compute_confusion(class_i, result)
        error -= tp
        print('\n------------------------------------------------------------\n')
        print(f'Confusion Matrix {class_i}')
        print(f'\t\tPredict number {class_i}\tPredict not number {class_i}')
        print(f'Is number {class_i}\t\t{tp}\t\t\t{fn}')
        print(f"Isn't number {class_i}\t\t{fp}\t\t\t{tn}")
        print(f'\nSensitivity (Successfully predict number {class_i}): {float(tp) / (tp + fn):.5f}')
        print(f'Specificity (Successfully predict not number {class_i}): {float(tn) / (fp + tn):.5f}')
    
    
    print(f'\nTotal iteration to converge: {count}')
    print(f'Total error rate {float(error) / len(bin_images):.16f}')

@jit
def e_step(lam, probability, x_train):
    data_num = len(x_train)
    pixel_num = len(x_train[0])
    
    ### need to return the new responsibility
    new_responsibility = np.zeros((data_num, 10))
    
    for data in range(data_num):
        ### for each data, compute the responsibility of each class
        for class_num in range(10):
            new_responsibility[data, class_num] = lam[class_num]
            
            for pixel in range(pixel_num):
                if x_train[data, pixel]:
                    new_responsibility[data, class_num] *= probability[class_num, pixel]
                else:
                    new_responsibility[data, class_num] *= (1.0 - probability[class_num, pixel])
        
        total_res = np.sum(new_responsibility[data, :])
        
        if total_res:
            new_responsibility[data, :] /= total_res
       
    return new_responsibility


@jit
def m_step(responsibility, x_train):
    pixel_num = len(x_train[0])
    
    sum_responsibility = np.zeros(10)
    for class_num in range(10):
        sum_responsibility[class_num] += np.sum(responsibility[:, class_num])
    
    ### the new probabilities will return
    probability = np.zeros((10, pixel_num))
    lam = np.zeros(10)
    
    for class_i in range(10):
        for pixel in range(pixel_num):
            for data in range(len(x_train)):
                probability[class_i, pixel] += responsibility[data, class_i] * x_train[data, pixel]
                
            probability[class_i, pixel] = (probability[class_i, pixel] + 1e-9) / (sum_responsibility[class_i] + 1e-9*pixel)
            
        lam[class_i] = (sum_responsibility[class_i] + 1e-9) / (np.sum(sum_responsibility) + 1e-9*10)
        
    return lam, probability
 
    
@jit
def find_matching(lam, probability, x_train, y_train):
    data_num = len(x_train)
    pixel_num = len(x_train[0])
    ### count each class of unknown classification
    ### row -> unknown classification
    ### col -> real class
    count = np.zeros((10, 10))
    
    ## the probability of each class
    result = np.zeros(10)
    
    for data in range(data_num):
        for class_i in range(10):
            result[class_i] = lam[class_i]
            
            for pixel in range(pixel_num):
                if x_train[data, pixel]:
                    result[class_i] *= probability[class_i, pixel]
                else:
                    result[class_i] *= (1.0 - probability[class_i, pixel])
        
        
        unknown_class = np.argmax(result)
        
        count[unknown_class, y_train[data]] += 1
    
    ### It will be a map between truth classes and virtual classes
    matching = np.full(10, -1, dtype=int)
    
    for _ in range(10):
        idx = np.unravel_index(np.argmax(count), (10, 10))
        matching[idx[0]] = idx[1]
        
        for k in range(10):
            count[idx[0]][k] = -1
            count[k][idx[1]] = -1
    
    return matching

@jit
def predict(lam, probability, x_train, y_train, matching):
    data_num = len(x_train)
    pixel_num = len(x_train[0])
    
    prediction = np.zeros((10, 10))
    result = np.zeros(10)
    
    for data in range(data_num):
        ### find the predict class
        for class_i in range(10):
            result[class_i] = lam[class_i]
            for pixel in range(pixel_num):
                if x_train[data, pixel]:
                    result[class_i] *= probability[class_i, pixel]
                else:
                    result[class_i] *= (1.0 - probability[class_i, pixel])
                    
        result_class = np.argmax(result)
        
        ### Increment the count of result class to real class, be sure it is mapped to the real class
        prediction[matching[result_class], y_train[data]] += 1
        
    return prediction


@jit
def compute_confusion(class_num, result):
    tp, fp, tn, fn = 0, 0, 0, 0
    for prediction in range(10):
        for real in range(10):
            if prediction == class_num and real == class_num:
                tp += result[prediction, real]
            elif prediction == class_num:
                fp += result[prediction, real]
            elif real == class_num:
                fn += result[prediction, real]
            else:
                tn += result[prediction, real]
                
    return int(tp), int(fp), int(tn), int(fn)

def b2i(b):
    return int(codecs.encode(b, 'hex'), 16)

        
if __name__ == '__main__':
    ## load data
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

    em_algo(x_train, y_train, train_row, train_col)