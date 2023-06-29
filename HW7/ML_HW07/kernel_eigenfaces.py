import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from PIL import Image
from scipy.spatial.distance import cdist

def pricipal_components_analysis(train_images, train_labels, test_images, test_labels, mode, k_neighbors, kernel_type, gamma):
    ### get number of images
    n_train = len(train_images)
    
    ### choose simple PCA or kernel PCA
    if mode == 'simple':
        ### PCA
        matrix = simple_pca(n_train, train_images)
    elif mode == 'kernel':
        ### Kernel PCA
        matrix = kernel_pca(train_images, kernel_type, gamma)
    else:
        raise BaseException(f'Invalid mode. The mode should be simple or kernel')
        
    ### find the biggest 25 eigenvectors
    target_eigenvectors = find_eigenvector(matrix)
    
    ### transform eigenvectors to eigenface
    eigenface(target_eigenvectors, 0)
    
    ### randomly recontruct 10 eigenface
    construct_face(n_train, train_images, target_eigenvectors)
    
    ### classify
    classify(n_train, len(test_images), train_images, train_labels, test_images, test_labels, target_eigenvectors, k_neighbors)
    
    ### show the result
    plt.tight_layout()
    plt.show()
    
    
def simple_pca(n_image, images):
    ### compute variance
    image_transpose = images.T
    mean = np.mean(image_transpose, axis=1)
    mean = np.tile(mean.T, (n_image, 1)).T
    difference = image_transpose - mean
    covariance = difference.dot(difference.T) / n_image
    
    return covariance
    

def kernel_pca(images, kernel_type, gamma):
    ### linear kernel:0 or RBF kernel:1
    if kernel_type == 'linear':
        ### linear kernel
        kernel = images.T.dot(images)
    elif kernel_type == 'rbf':
        ### RBF kernel
        kernel = np.exp(-gamma * cdist(images.T, images.T, 'sqeuclidean'))
    else:
        raise BaseException(f'Invalid kernel type. The kernel type should be linear or rbf')
        
    ### get centered kernel
    matrix_n = np.ones((29 * 24, 29 * 24), dtype=float) / (29 * 24)
    matrix = kernel - matrix_n.dot(kernel) - kernel.dot(matrix_n) + matrix_n.dot(kernel).dot(matrix_n)
    
    return matrix
    
    

def linear_discriminative_analysis(train_images, train_labels, test_images, test_labels, mode, k_neighbors, kernel_type, gamma):
    ### get number of images and number of classes
    n_train = len(train_images)
    _, num_of_each_class = np.unique(train_labels, return_counts=True)
    
    ### choose simple LDA or kernel LDA
    if mode == 'simple':
        ### LDA
        matrix = simple_lda(num_of_each_class, train_images, train_labels)
    elif mode == 'kernel':
        ### Kernel LDA
        matrix = kernel_lda(num_of_each_class, train_images, train_labels, kernel_type, gamma)
    else:
        raise BaseException(f'Invalid mode. The mode should be simple or kernel')
        
    ### find the biggest 25 eigenvectors
    target_eigenvectors = find_eigenvector(matrix)
    
    ### transform eigenvectors to eigenface
    eigenface(target_eigenvectors, 1)
    
    ### randomly recontruct 10 eigenface
    construct_face(n_train, train_images, target_eigenvectors)
    
    ### classify
    classify(n_train, len(test_images), train_images, train_labels, test_images, test_labels, target_eigenvectors, k_neighbors)
    
    ### show the result
    plt.tight_layout()
    plt.show()
    
    

def simple_lda(num_of_each_class, images, labels):
    ### get overall mean
    overall_mean = np.mean(images, axis=0)
    
    ### mean of each class
    n_class = len(num_of_each_class)
    class_mean = np.zeros((n_class, 29 * 24))
    for label in range(n_class):
        class_mean[label, :] = np.mean(images[labels == label + 1], axis=0)
        
    ### get between class scatter
    scatter_b = np.zeros((29 * 24, 29 * 24), dtype=float)
    for idx, num in enumerate(num_of_each_class):
        difference = (class_mean[idx] - overall_mean).reshape((29 * 24, 1))
        scatter_b += num * difference.dot(difference.T)
        
    ### get within class scatter
    scatter_w = np.zeros((29 * 24, 29 * 24), dtype=float)
    for idx, mean in enumerate(class_mean):
        difference = images[labels == idx + 1] - mean
        scatter_w += difference.T.dot(difference)
        
    ### get Sw^(-1) * Sb
    matrix = np.linalg.pinv(scatter_w).dot(scatter_b)
    
    return matrix
    

def kernel_lda(num_of_each_class, images, labels, kernel_type, gamma):
    n_class = len(num_of_each_class)
    n_image = len(images)
    
    if kernel_type == 'linear':
        ### Linear
        kernel_of_each_class = np.zeros((n_class, 29 * 24, 29 * 24))
        for idx in range(n_class):
            image = images[labels == idx + 1]
            kernel_of_each_class[idx] = image.T.dot(image)
        kernel_of_all = images.T.dot(images)
    elif kernel_type == 'rbf':
        ### RBF
        kernel_of_each_class = np.zeros((n_class, 29 * 24, 29 * 24))
        for idx in range(n_class):
            image = images[labels == idx + 1]
            kernel_of_each_class[idx] = np.exp(-gamma * cdist(image.T, image.T, 'sqeuclidean'))
        kernel_of_all = np.exp(-gamma * cdist(images.T, images.T, 'sqeuclidean'))
    else:
        raise BaseException(f'Invalid kernel type. The kernel type should be linear or rbf')
    
    ### compute matrix_n
    matrix_n = np.zeros((29 * 24, 29 * 24))
    identity_matrix = np.eye(29 * 24)
    for idx, num in enumerate(num_of_each_class):
        matrix_n += kernel_of_each_class[idx].dot(identity_matrix - num * identity_matrix).dot(kernel_of_each_class[idx].T)
        
    ### coompute matrix_m
    matrix_m_i = np.zeros((n_class, 29 * 24))
    for idx, kernel in enumerate(kernel_of_each_class):
        for row_idx, row in enumerate(kernel):
            matrix_m_i[idx, row_idx] = np.sum(row) / num_of_each_class[idx]
            
    matrix_m_star = np.zeros(29 * 24)
    for idx, row in enumerate(kernel_of_all):
        matrix_m_star[idx] = np.sum(row) / n_image
    
    matrix_m = np.zeros((29 * 24, 29 * 24))
    for idx, num in enumerate(num_of_each_class):
        difference = (matrix_m_i[idx] - matrix_m_star).reshape((29 * 24, 1))
        matrix_m += num * difference.dot(difference.T)
        
    ### get N^(-1) * M
    matrix = np.linalg.pinv(matrix_n).dot(matrix_m)
    
    return matrix
    
    
    
def find_eigenvector(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    ### find the biggest 25 eigenvectors
    target_idx = np.argsort(eigenvalues)[::-1][:25]
    target_eigenvectors = eigenvectors[:, target_idx].real
    
    return target_eigenvectors
    
    
    

def eigenface(eigenvectors, mode):
    ### plot the eigenface of PCA:0 or LDA:1
    faces = eigenvectors.T.reshape((25, 29, 24))
    fig = plt.figure(1)
    fig.canvas.set_window_title(f'{"Eigenfaces" if mode == 0 else "Fisherfaces"}')
    for idx in range(25):
        plt.subplot(5, 5, idx+1)
        plt.axis('off')
        plt.imshow(faces[idx, :, :], cmap='gray')
    
    
    
def construct_face(n_image, images, eigenvectors):
    ### construct 10 eigenfaces randomly
    
    reconstruct_image = np.zeros((10, 29 * 24))
    
    choice = np.random.choice(n_image, 10)
    for idx in range(10):
        reconstruct_image[idx, :] = images[choice[idx], :].dot(eigenvectors).dot(eigenvectors.T)
        
    fig = plt.figure(2)
    fig.canvas.set_window_title(f'Reconstructed faces')
    for idx in range(10):
        ### original image
        plt.subplot(10, 2, idx * 2 + 1)
        plt.axis('off')
        plt.imshow(images[choice[idx], :].reshape(29, 24), cmap='gray')
        
        ### reconstructed image
        plt.subplot(10, 2, idx * 2 + 2)
        plt.axis('off')
        plt.imshow(reconstruct_image[idx, :].reshape(29, 24), cmap='gray')
    
    

def decorrelate(n_image, images, eigenvectors):
    ### decorrelate original images to components space
    decorrlated_image = np.zeros((n_image, 25))
    
    for idx, image in enumerate(images):
        decorrlated_image[idx, :] = image.dot(eigenvectors)
        
    return decorrlated_image

    
def classify(n_train, n_test, train_images, train_labels, test_images, test_labels, eigenvectors, k_neighbors):
    decorrelate_train = decorrelate(n_train, train_images, eigenvectors)
    decorrelate_test = decorrelate(n_test, test_images, eigenvectors)
    
    error = 0
    
    distance = np.zeros(n_train)
    
    for test_idx, test_image in enumerate(decorrelate_test):
        for train_idx, train_image in enumerate(decorrelate_train):
            distance[train_idx] = np.linalg.norm(test_image - train_image)
            
        min_distance = np.argsort(distance)[:k_neighbors]
        
        predict = np.argmax(np.bincount(train_labels[min_distance]))
        if predict != test_labels[test_idx]:
            error += 1
    print(f'Error count: {error}\nError rate: {float(error) / n_test}')
    
    
    
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--algo', help='PCA, LDA', type=str)
    parser.add_argument('--mode', help='simple,kernel', type=str)
    parser.add_argument('--k_neighbors', default=5, type=int)
    parser.add_argument('--type_kernel', help='linear, rbf', type=str)
    parser.add_argument('--gamma', default=0.000001, type=float)
    
    return parser.parse_args()
    
    
    
if __name__ == '__main__':
    args = parse_arguments()
    dirname = './Yale_Face_Database'
    algo = args.algo
    mode = args.mode
    k_neighbors = args.k_neighbors
    kernel_type = args.type_kernel
    gamma = args.gamma
    
    ### read training images
    train_images, train_labels = None, None
    n_files = 0
    with os.scandir(f'{dirname}/Training') as directory:
        ### get number of files
        n_files = len([file for file in directory if file.is_file()])
        
    with os.scandir(f'{dirname}/Training') as directory:
        train_labels = np.zeros(n_files, dtype=int)
        ### image: 29 * 24
        train_images = np.zeros((n_files, 29 * 24))
        for idx, file in enumerate(directory):
            if file.path.endswith('.pgm') and file.is_file():
                face = np.asarray(Image.open(file.path).resize((24, 29))).reshape(1, -1)
                train_images[idx, :] = face
                train_labels[idx] = int(file.name[7:9])
                
    ### read testing images
    test_images, test_labels = None, None
    n_files = 0
    with os.scandir(f'{dirname}/Testing') as directory:
        ### get number of files
        n_files = len([file for file in directory if file.is_file()])
        
    with os.scandir(f'{dirname}/Testing') as directory:
        test_labels = np.zeros(n_files, dtype=int)
        ### image: 29 * 24
        test_images = np.zeros((n_files, 29 * 24))
        for idx, file in enumerate(directory):
            if file.path.endswith('.pgm') and file.is_file():
                face = np.asarray(Image.open(file.path).resize((24, 29))).reshape(1, -1)
                test_images[idx, :] = face
                test_labels[idx] = int(file.name[7:9])
                
    if algo == 'PCA':
        ### PCA
        pricipal_components_analysis(train_images, train_labels, test_images, test_labels, mode, k_neighbors, kernel_type, gamma)
    else:
        ### LDA
        linear_discriminative_analysis(train_images, train_labels, test_images, test_labels, mode, k_neighbors, kernel_type, gamma)