import argparse
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
from kmeans import capture_current_state, compute_kernel
import numpy as np

def spectral_clustering(n_rows, n_cols, n_clusters, matrix_u, mode, cut, index):
    centers = init_centers(n_rows, n_cols, n_clusters, matrix_u, mode)
    
    ### k-means
    clusters = kmeans(n_rows, n_cols, n_clusters, matrix_u, centers, index, mode, cut)
    
    ### plot data point in eigenspace if number of clusters is 2
    if n_clusters == 2:
        plot_result(matrix_u, clusters, index, mode, cut)
        

    
def compute_matrix_u(matrix_w, cut, n_clusters):
    ### get the laplacian matrix L and degree matrix D
    matrix_d = np.zeros_like(matrix_w)
    
    for idx, row in enumerate(matrix_w):
        matrix_d[idx, idx] += np.sum(row)
        
    matrix_l = matrix_d - matrix_w
    
    if cut:
        ### normalized cut
        ### compute the normalized laplacian
        for idx in range(len(matrix_d)):
            matrix_d[idx, idx] = 1.0 / np.sqrt(matrix_d[idx, idx])
        
        matrix_l = matrix_d.dot(matrix_l).dot(matrix_d)
        
    ### else is the ratio cut
    ### get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix_l)
    eigenvectors = eigenvectors.T
    
    ### sort eigenvalues and find indices of nonzero eigenvalues
    sort_idx = np.argsort(eigenvalues)
    sort_idx = sort_idx[eigenvalues[sort_idx] > 0]
    
    return eigenvectors[sort_idx[:n_clusters]].T
    
def init_centers(n_rows, n_cols, n_clusters, matrix_u, mode):
    if not mode:
        return matrix_u[np.random.choice(n_rows * n_cols, n_clusters)]
    else:
        grid = np.indices((n_rows, n_cols))
        row_indices, col_indices = grid[0], grid[1]
        
        indices = np.hstack((row_indices.reshape(-1, 1), col_indices.reshape(-1, 1)))
        
        n_points = n_rows * n_cols
        centers = [indices[np.random.choice(n_points, 1)[0]].tolist()]
        
        for _ in range(n_clusters - 1):
            distance = np.zeros(n_points)
            for idx, point in enumerate(indices):
                min_distance = np.Inf
                for center in centers:
                    dist = np.linalg.norm(point - center)
                    min_distance = dist if dist < min_distance else min_distance
                distance[idx] = min_distance
                
            distance /= np.sum(distance)
            centers.append(indices[np.random.choice(n_points, 1, p=distance)[0]].tolist())
            
        
        ### change from index to feature index
        for idx, center in enumerate(centers):
            centers[idx] = matrix_u[center[0] * n_rows + center[1], :]
            
        return np.array(centers)
    
    
    
def kmeans(n_rows, n_cols, n_clusters, matrix_u, centers, index, mode, cut):
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    
    if n_clusters > 3:
        colors = np.append(color, np.random.choice(256, (n_clusters - 3, 3)), axis=0)
        
    n_points = n_rows * n_cols
    img = []
    
    ### k means
    current_centers = centers.copy()
    new_cluster = np.zeros(n_points, dtype=int)
    count = 0
    iteration = 100
    while True:
        ### get new cluster
        new_cluster = kmeans_clustering(n_points, n_clusters, matrix_u, current_centers)
        
        ### get new centers
        new_centers = kmeans_recompute_centers(n_clusters, matrix_u, new_cluster)
        
        ### capture the new state
        img.append(capture_current_state(n_rows, n_cols, new_cluster, colors))
        
        if np.linalg.norm((new_centers - current_centers), ord=2) < 0.01 or count >= iteration:
            break
            
        current_centers = new_centers.copy()
        count += 1
        
    ### save as gif
    filename = f'./gifs/spectral_clustering/image{index}_cluster{n_clusters}_{"kmeans" if mode else "random"}_{"normalized" if cut else "ratio"}.gif'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if len(img) > 1:
        img[0].save(filename, save_all=True, append_images=img[1:], optimize=False, loop=0, duration=100)
    else:
        img[0].save(filename)
        
    return new_cluster
    
    
    
def kmeans_clustering(n_points, n_clusters, matrix_u, centers):
    new_clusters = np.zeros(n_points, dtype=int)
    
    for p in range(n_points):
        distance = np.zeros(n_clusters)
        for idx, center in enumerate(centers):
            distance[idx] = np.linalg.norm((matrix_u[p] - center), ord=2)
            
        new_clusters[p] = np.argmin(distance)
        
    return new_clusters
    
    
    
def kmeans_recompute_centers(n_clusters, matrix_u, current_cluster):
    new_centers = []
    for cluster in range(n_clusters):
        points_in_cluster = matrix_u[current_cluster == cluster]
        new_center = np.average(points_in_cluster, axis=0)
        new_centers.append(new_center)
        
    return np.array(new_centers)
    
    
def plot_result(matrix_u, clusters, index, mode, cut):
    color = ['r', 'b']
    plt.clf()
    
    for idx, point in enumerate(matrix_u):
        plt.scatter(point[0], point[1], c=color[clusters[idx]])
        
    ### save the plot
    filename = f'./gifs/spectral_clustering/eigenspace{index}_{"kmean" if mode else "random"}_{"normalized" if cut else "ratio"}.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.savefig(filename)
    
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Spectral clustering')
    parser.add_argument('--n_cluster', default=2, type=int)
    parser.add_argument('--mode', default=0, type=int)
    parser.add_argument('--cut', default=0, type=int)
    
    return parser.parse_args()
    
    
if __name__ == '__main__':
    args = parse_arguments()
    mode = args.mode
    n_cluster = args.n_cluster
    cut = args.cut
    gamma_s = 0.0001
    gamma_c = 0.001
    
    images = [Image.open('image1.png'), Image.open('image2.png')]
    images[0] = np.asarray(images[0])
    images[1] = np.asarray(images[1])
    
    for i, image in enumerate(images):
        rows, cols, _ = image.shape
        gram_matrix = compute_kernel(image, gamma_s, gamma_c)
        
        matrix_u = compute_matrix_u(gram_matrix, cut, n_cluster)
        
        if cut:
            ### normalized cut
            sum_of_each_row = np.sum(matrix_u, axis=1)
            for j in range(len(matrix_u)):
                matrix_u[j, :] /= sum_of_each_row[j]
                
        spectral_clustering(rows, cols, n_cluster, matrix_u, mode, cut, i)