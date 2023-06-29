import argparse
import os 
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist

def kernel_kmeans(n_rows, n_cols, n_clusters, cluster, kernel, mode, index):
    ### colors
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    if n_clusters > 3:
        colors = np.append(colors, np.random.choice(256, (n_clusters - 3, 3)), axis=0)
        
    ### list storing image of cluster state
    img = [capture_current_state(n_rows, n_cols, cluster, colors)]
    
    ### kernel kmeans
    current_cluster = cluster.copy()
    count = 0
    iteration = 100
    while True:
        ### get new cluster
        new_cluster = kernel_clustering(n_rows * n_cols, n_clusters, kernel, current_cluster)
        
        ### capture new state
        img.append(capture_current_state(n_rows, n_cols, new_cluster, colors))
        
        if np.linalg.norm((new_cluster - current_cluster), ord=2) < 0.001 or count >= iteration:
            break
            
        current_cluster = new_cluster.copy()
        count += 1
    
    ### save as gif
    filename = f'./gifs/kernel_kmeans/image{index}_cluster{n_clusters}_{"kmeans" if mode else "random"}.gif'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img[0].save(filename, save_all=True, append_images=img[1:], optimize=False, loop=0, duration=100)
    
    
def kernel_clustering(n_points, n_clusters, kernel, cluster):
    ### number of members in each cluster
    n_members = np.array([np.sum(np.where(cluster == c, 1, 0)) for c in range(n_clusters)])
    
    ### sum of pairwise kernel distance of each cluster
    pairwise_distance = get_sum_of_pairwise_distance(n_points, n_clusters, n_members, kernel, cluster)
    
    new_cluster = np.zeros(n_points, dtype=int)
    for p in range(n_points):
        distance = np.zeros(n_clusters)
        for c in range(n_clusters):
            distance[c] += kernel[p, p] + pairwise_distance[c]
            
            ### the distance between others in the target cluster
            distance2others = np.sum(kernel[p, :][np.where(cluster == c)])
            distance[c] -= 2.0 / n_members[c] * distance2others
        new_cluster[p] = np.argmin(distance)
        
    return new_cluster
    
def get_sum_of_pairwise_distance(n_points, n_clusters, n_members, kernel, cluster):
    pairwise_distance = np.zeros(n_clusters)
    for c in range(n_clusters):
        tmp_kernel =  kernel.copy()
        for p in range(n_points):
            if cluster[p] != c:
                tmp_kernel[p, :]
                tmp_kernel[:, p]
        pairwise_distance = np.sum(tmp_kernel)
        
    ### if n_members == 0
    n_members[n_members == 0] = 1
    
    return pairwise_distance / n_members ** 2

def capture_current_state(n_rows, n_cols, cluster, colors):
    state = np.zeros((n_rows * n_cols, 3))
    for p in range(n_rows * n_cols):
        state[p, :] = colors[cluster[p], :]
        
    state = state.reshape((n_rows, n_cols, 3))
    
    return Image.fromarray(np.uint8(state))
    
def init_clustering(n_rows, n_cols, n_clusters, kernel, mode):
    ### init centers
    centers = choose_center(n_rows, n_cols, n_clusters, mode)
    
    ### k-means
    n_points = n_rows * n_cols
    cluster = np.zeros(n_points, dtype=int)
    for p in range(n_points):
        ### calculate the distance between each center and each point
        distance = np.zeros(n_clusters)
        for idx, center in enumerate(centers):
            seq_center = center[0] * n_rows + center[1]
            distance[idx] = kernel[p, p] + kernel[seq_center, seq_center] - 2 * kernel[p, seq_center]
            
        ### put the point into the nearest cluster
        cluster[p] = np.argmin(distance)
        
    return cluster
    
def choose_center(n_rows, n_cols, n_clusters, mode):
    if not mode:
        ### random strategy
        return np.random.choice(100, (n_clusters, 2))
    else:
        ### kmean++ strategy
        
        grid = np.indices((n_rows, n_cols))
        row_indices, col_indices = grid[0], grid[1]
        indices = np.hstack((row_indices.reshape(-1, 1), col_indices.reshape(-1, 1)))
        
        ### pick the init center randomly
        n_points = n_rows * n_cols
        centers = [indices[np.random.choice(n_points, 1)[0]].tolist()]
        
        ### find remaining centers
        for _ in range(n_clusters - 1):
            distance = np.zeros(n_points)
            for idx, point in enumerate(indices):
                min_distance = np.Inf
                for center in centers:
                    dist = np.linalg.norm(point - center)
                    min_distance = dist if dist < min_distance else min_distance
                distance[idx] = min_distance
            ### get the probability of the distance
            distance /= np.sum(distance)
            ### new center
            centers.append(indices[np.random.choice(n_points, 1, p=distance)[0]].tolist())
            
        return np.array(centers)


def compute_kernel(image, gamma_s, gamma_c):
    row, col, color = image.shape
    ### the distance of each color
    color_distance = cdist(image.reshape(row * col, color), image.reshape(row * col, color), 'sqeuclidean')
    ### the indices vector
    grid = np.indices((row, col))
    row_indices, col_indices = grid[0], grid[1]
    indices = np.hstack((row_indices.reshape(-1, 1), col_indices.reshape(-1, 1)))
    
    ### the spatial distance
    spatial_distance = cdist(indices, indices, 'sqeuclidean')
    
    return np.multiply(np.exp(-gamma_s * spatial_distance), np.exp(-gamma_c * color_distance))


def parse_arguments():
    parser = argparse.ArgumentParser(description='kernel k-means')
    parser.add_argument('--n_cluster', default=3, type=int)
    parser.add_argument('--mode', default=0, type=int)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    images = [Image.open('image1.png'), Image.open('image2.png')]
    mode = args.mode
    n_cluster = args.n_cluster
    gamma_s = 0.0001
    gamma_c = 0.001
    
    images[0] = np.asarray(images[0])
    images[1] = np.asarray(images[1])
    
    for i, image in enumerate(images):
        gram_matrix = compute_kernel(image, gamma_s, gamma_c)
        
        rows, cols, _ = image.shape
        clusters = init_clustering(rows, cols, n_cluster, gram_matrix, mode)
        kernel_kmeans(rows, cols, n_cluster, clusters, gram_matrix, mode, i)