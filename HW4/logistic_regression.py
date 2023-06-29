import argparse
import numpy as np
from scipy.special import expit, expm1
from scipy.linalg import inv, pinv
import matplotlib.pyplot as plt

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


def gradient_descent(phi, group):
    ### initail weight
    omega = np.random.rand(3, 1)
    
    count = 0
    while True:
        count += 1
        old_omega = omega.copy()
        omega += phi.T.dot(group - expit(phi.dot(omega)))
        
        if np.linalg.norm(omega - old_omega) < 0.0001 or count > 1000:
            break
    
    return omega
   
def newton_method(N, phi, group):
    ### initial weight
    omega = np.random.rand(3, 1)
    
    ### used to make the hessian matrix
    d = np.zeros((N*2, N*2))
    
    count = 0
    while True:
        count += 1
        old_omega = omega.copy()
        
        
        product = phi.dot(omega)
        diag = (expm1(-product) + 1) * np.power(expit(product), 2)
        np.fill_diagonal(d, diag)
        
        ### make the hessian matrix
        hessian = phi.T.dot(d.dot(phi))
        
        ### update
        try:
            omega += inv(hessian).dot(phi.T.dot(group - expit(phi.dot(omega))))
        except:
            omega += phi.T.dot(group - expit(phi.dot(omega)))
            # omega += pinv(hessian).dot(phi.T.dot(group - expit(phi.dot(omega))))
        
        if np.linalg.norm(omega - old_omega) < 0.0001 or count > 1000:
            break
        
    return omega

def logistic_regression(N, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2):
    ### generate datas
    d1 = np.zeros((N, 2))
    d2 = np.zeros((N, 2))
    
    for i in range(N):
        d1[i, 0], d1[i, 1] = univariate_gaussian_data_generator(mx1, vx1)
        d2[i, 0], d2[i, 1] = univariate_gaussian_data_generator(mx2, vx2)
        
    ### phi
    phi = np.ones((N*2, 3))
    phi[:N, :2] = d1
    phi[N:, :2] = d2
    
    ### group
    group = np.zeros((N*2, 1), dtype=int)
    group[N:, 0] = 1
    
    ### GD method
    GD_omega = gradient_descent(phi, group)
    
    ### Newton method
    newton_omega = newton_method(N, phi, group)
    
    ### show the result
    ### GD
    GD_confusion = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    GD_class1, GD_class2 = [], []
    
    for i in range(N * 2):
        if phi[i].dot(GD_omega) >= 0:
            ### in class 2
            GD_class2.append(list(phi[i, :2]))
            if group[i, 0] == 1:
                ### predict correctly
                GD_confusion['TP'] += 1
            else:
                ### incorrectly
                GD_confusion['FP'] += 1
        else:
            ### in class 1
            GD_class1.append(list(phi[i, :2]))
            if group[i, 0] == 0:
                ### predict correctly
                GD_confusion['TN'] += 1
            else:
                ### incorrectly
                GD_confusion['FN'] += 1
                
    GD_class1 = np.array(GD_class1)
    GD_class2 = np.array(GD_class2)
    
    ### newton
    newton_confusion = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    newton_class1, newton_class2 = [], []
    
    for i in range(N * 2):
        if phi[i].dot(newton_omega) >= 0:
            ### in class 2
            newton_class2.append(list(phi[i, :2]))
            if group[i, 0] == 1:
                ### predict correctly
                newton_confusion['TP'] += 1
            else:
                ### incorrectly
                newton_confusion['FP'] += 1
        else:
            ### in class 1
            newton_class1.append(list(phi[i, :2]))
            if group[i, 0] == 0:
                ### predict correctly
                newton_confusion['TN'] += 1
            else:
                ### incorrectly
                newton_confusion['FN'] += 1
                
    newton_class1 = np.array(newton_class1)
    newton_class2 = np.array(newton_class2)
    
    print('Gradient descent:\n')
    print('w:')
    for weight in GD_omega:
        print(f'{weight[0]:.10f}')
    print('\nConfusion Matrix:')
    print('\t\tPredict cluster 1\tPredict cluster 2')
    print(f'Is cluster 1\t\t{GD_confusion["TP"]}\t\t\t{GD_confusion["FN"]}')
    print(f'Is cluster 2\t\t{GD_confusion["FP"]}\t\t\t{GD_confusion["TN"]}')
    print(f'\nSensitivity (Successfully predict cluster 1): {GD_confusion["TP"] / (GD_confusion["TP"] + GD_confusion["FN"])}')
    print(f'\nSensitivity (Successfully predict cluster 2): {GD_confusion["TN"] / (GD_confusion["FP"] + GD_confusion["TN"])}')
    
    print('\n----------------------------------------------------------')
    
    print("Newton's method:\n")
    print('w:')
    for weight in newton_omega:
        print(f'{weight[0]:.10f}')
    print('\nConfusion Matrix:')
    print('\t\tPredict cluster 1\tPredict cluster 2')
    print(f'Is cluster 1\t\t{newton_confusion["TP"]}\t\t\t{newton_confusion["FN"]}')
    print(f'Is cluster 2\t\t{newton_confusion["FP"]}\t\t\t{newton_confusion["TN"]}')
    print(f'\nSensitivity (Successfully predict cluster 1): {newton_confusion["TP"] / (newton_confusion["TP"] + newton_confusion["FN"])}')
    print(f'\nSensitivity (Successfully predict cluster 2): {newton_confusion["TN"] / (newton_confusion["FP"] + newton_confusion["TN"])}')
    
    
    ### graph
    plt.subplot(131)
    plt.title('Ground truth')
    plt.scatter(phi[:N, 0], phi[:N, 1], color='r')
    plt.scatter(phi[N:, 0], phi[N:, 1], color='b')
    
    plt.subplot(132)
    plt.title('Gradient descent')
    if GD_class1.size:
        plt.scatter(GD_class1[:, 0], GD_class1[:, 1], color='r')
    if GD_class2.size:
        plt.scatter(GD_class2[:, 0], GD_class2[:, 1], color='b')
    
    plt.subplot(133)
    plt.title("Newton's method")
    if newton_class1.size:
        plt.scatter(newton_class1[:, 0], newton_class1[:, 1], color='r')
    if newton_class2.size:
        plt.scatter(newton_class2[:, 0], newton_class2[:, 1], color='b')
    
    plt.tight_layout()
    plt.show()
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Logistic Regression')
    parser.add_argument('--N', type=int)
    parser.add_argument('--mx1', type=float)
    parser.add_argument('--vx1', type=float)
    parser.add_argument('--my1', type=float)
    parser.add_argument('--vy1', type=float)
    parser.add_argument('--mx2', type=float)
    parser.add_argument('--vx2', type=float)
    parser.add_argument('--my2', type=float)
    parser.add_argument('--vy2', type=float)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    N = args.N
    mx1 = args.mx1
    vx1 = args.vx1
    my1 = args.my1
    vy1 = args.vy1
    mx2 = args.mx2
    vx2 = args.vx2
    my2 = args.my2
    vy2 = args.vy2
    
    logistic_regression(N, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2)