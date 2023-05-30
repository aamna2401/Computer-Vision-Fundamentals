import numpy as np
import matplotlib.pyplot as plt
from utils import *

def find_projection(pts2d, pts3d):
    """
    Computes camera projection matrix M that goes from world 3D coordinates
    to 2D image coordinates.

    [u v 1]^T === M [x y z 1]^T

    Where (u,v) are the 2D image coordinates and (x,y,z) are the world 3D
    coordinates

    Inputs:
    - pts2d: Numpy array of shape (N,2) giving 2D image coordinates
    - pts3d: Numpy array of shape (N,3) giving 3D world coordinates

    Returns:
    - P: Numpy array of shape (3,4) giving the camera projection matrix P

    """
    #pts2d = homogenize(pts2d)
    #pts3d = homogenize(pts3d)
   
    ###########################################################################
    # TODO: Your code here                                                    
    ###########################################################################
    ###########################################################################
    assert pts3d.shape[0] == pts2d.shape[0]
    no_of_points = pts3d.shape[0]
    M = np.ones(shape=(2*no_of_points, 12))
    
    count = 0
    
    for i in range(no_of_points):
        
        w = pts3d[i, :]
        p = pts2d[i, :]
        
        X, Y, Z = w
        u, v = p
        rows = np.zeros(shape=(2, 12))
        
        rows[0, 0], rows[0, 1], rows[0, 2], rows[0, 3] = X, Y, Z, 1
        rows[0, 8], rows[0, 9], rows[0, 10], rows[0, 11] = -u * X, -u * Y, -u * Z, -u

        rows[1, 4], rows[1, 5], rows[1, 6], rows[1, 7] = X, Y, Z, 1
        rows[1, 8], rows[1, 9], rows[1, 10], rows[1, 11] = -v * X, -v * Y, -v * Z, -v
        
        M[count:count+2, :] = rows
        count += 2
                
    
    # compute At x A
    M_ = np.matmul(M.T, M)
    # compute its eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(M_)
    # find the eigenvector with the minimum eigenvalue
    # (numpy already returns sorted eigenvectors wrt their eigenvalues)
    m = eigenvectors[:, 11]
    
    # reshape m back to a matrix
    P = m.reshape(3, 4)
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return P

def predicted_points(pts3d, P):
    #get homogenous coordinate
    pts3d = pts3d.T
    homo_points = np.vstack((pts3d, np.ones(pts3d.shape[1])))  
    #multiply projection matrix with pts3d  
    homo_points_i = P @  homo_points
    #get 2d points by dividing last element
    homo_points_i[0, :] = homo_points_i[0, :] / homo_points_i[2, :]
    homo_points_i[1, :] = homo_points_i[1, :] / homo_points_i[2, :]
    pred_points = homo_points_i[:2, :]    
    return pred_points

def draw_comparison(pts2d, pred_points):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    no_of_points = pts2d.shape[0]
    for i in range(no_of_points):
        if i == 0:
            o_label = "groundtruth"
            g_label = "predictions"
        else:
            o_label = ""
            g_label = ""
            
        ax.scatter(*pts2d.reshape(-1, 2)[i], color="orange", alpha=0.75, label=o_label)
        ax.scatter(*pred_points.reshape(-1, 2)[i], color="green", alpha=0.75, label=g_label)
        
    ax.set_title("Actual 2D Points vs predictions")
    ax.legend()
    
def l2_norm(pts2d, predicted_points):
    l2 = np.sum(np.power((pts2d.T-pred_points),2))
    return l2
if __name__ == '__main__':
    pts2d = np.loadtxt("task1/pts2d.txt")
    pts3d = np.loadtxt("task1/pts3d.txt")

    # Alternately, for some of the data, we provide pts1/pts1_3D, which you
    # can check your system on via
    """
    data = np.load("task23/ztrans/data.npz")
    pts2d = data['pts1']
    pts3d = data['pts1_3D']
    """
    #Find Projection Matrix
    P = find_projection(pts2d,pts3d)
    #Find Predicted Points
    pred_points = predicted_points(pts3d, P)
    #draw predictions
    draw_comparison(pts2d, pred_points)
    #Find L2 norm 
    l2 = l2_norm(pts2d, predicted_points)
    
