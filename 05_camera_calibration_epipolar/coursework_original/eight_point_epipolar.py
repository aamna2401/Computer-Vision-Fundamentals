from utils import dehomogenize, homogenize, draw_epipolar, visualize_pcd
import numpy as np
import scipy
import cv2 
import os
import matplotlib.pyplot as plt


#(1/N(summation(xi-mean)^2))^0.5
def Normalize(pts1):
    mean_shift = np.zeros(pts1.shape)
    mean0 = np.mean(pts1[:,0])
    mean1 = np.mean(pts1[:,1])
    mean_shift[:,0] = pts1[:,0] - mean0
    mean_shift[:,1] = pts1[:,1] - mean1
    N = len(pts1)
    s1 = (N*2)**0.5/(sum([((x-mean1)**2 + (y-mean1)**2)**0.5 for x,y in mean_shift]))
    T1 = np.array([[s1, 0, -mean0*s1], [0, s1, -mean1*s1], [0, 0, 1]])
    #T2 = np.array([[s2, 0, -mean2[0]*s2], [0, s2, -mean2[1]*s2], [0, 0, 1]])
    return T1
       
    
def find_fundamental_matrix(shape, pts1, pts2):
    """
    Computes Fundamental Matrix F that relates points in two images by the:

        [u' v' 1] F [u v 1]^T = 0
        or
        l = F [u v 1]^T  -- the epipolar line for point [u v] in image 2
        [u' v' 1] F = l'   -- the epipolar line for point [u' v'] in image 1

    Where (u,v) and (u',v') are the 2D image coordinates of the left and
    the right images respectively.

    Inputs:
    - shape: Tuple containing shape of img1
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    """
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    #Normalize data
    T1 = Normalize(pts1)
    T2 = Normalize(pts2)
    #get homogenous coordinate
    #pts1_ho = np.hstack([pts1, np.ones((pts1.shape[0], 1))])  
    pts1_ho = homogenize(pts1)
    pts2_ho = homogenize(pts2)
    #multiply projection matrix with pts3d  
    pts1 = T1 @  pts1_ho.T
    pts2 = T2 @ pts2_ho.T
    #normalized points
    u = pts1.T
    v = pts2.T
    
    n = u.shape[0]
    if v.shape[0] != n:
        raise ValueError("Number of points don't match.")
    A = np.zeros((len(u),9))
    # img1 and img2
    for i in range(len(u)):
        A[:,0] = u[:,0]*v[:,0]
        A[:,1] = u[:,0]*v[:,1]
        A[:,2] = u[:,0]
        A[:,3] = u[:,1]*v[:,0]
        A[:,4] = u[:,1]*v[:,1]
        A[:,5] = u[:,1] 
        A[:,6] = u[:,0]
        A[:,7] = u[:,1]
        A[:,8] = 1.0 
        
    # compute linear least square solution
    U,Su,V = np.linalg.svd(A)
    F_hat = V[-1].reshape(3,3)
    # Enforce rank contraint = 2 
    u,su,v = np.linalg.svd(F_hat)
    #dot product
    f_hat = u @ np.diag([*su[:2], 0]) @ v
    #transformation
    F = f_hat/f_hat[2,2]
    #restore 
    Fmat = T2.transpose() @ (np.dot(F, T1))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return T1,T2, F, Fmat


def compute_epipoles(F):
    """
    Given a Fundamental Matrix F, return the epipoles represented in
    homogeneous coordinates.

    Check: e2@F and F@e1 should be close to [0,0,0]

    Inputs:
    - F: the fundamental matrix

    Return:
    - e1: the epipole for image 1 in homogeneous coordinates
    - e2: the epipole for image 2 in homogeneous coordinates
    """
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    F = F.transpose() @ F
    w3, v3 = np.linalg.eig(F)
    small = np.argmin(w3)
    F_vect = v3[:,small]
    e1 = F_vect/F_vect[2]

    FT_mat = F @ F.transpose()
    w4 , v4 = np.linalg.eig(FT_mat)
    small = np.argmin(w4)
    F_vect = v4[:,small]
    e2 = F_vect/F_vect[2]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return e1,e2


def find_triangulation(K1, K2, F, pts1, pts2):
    """
    Extracts 3D points from 2D points and camera matrices. Let X be a
    point in 3D in homogeneous coordinates. For two cameras, we have

        p1 === M1 X
        p2 === M2 X

    Triangulation is to solve for X given p1, p2, M1, M2.

    Inputs:
    - K1: Numpy array of shape (3,3) giving camera instrinsic matrix for img1
    - K2: Numpy array of shape (3,3) giving camera instrinsic matrix for img2
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - pcd: Numpy array of shape (N,4) giving the homogeneous 3D point cloud
      data
    """
    pcd = None
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pcd


if __name__ == '__main__':
    # You can run it on one or all the examples
    names = os.listdir("task23")
    output = "results/"

    if not os.path.exists(output):
        os.mkdir(output)
    Fund_mat = []
    for name in names:
        name = 'xtrans'
        # load the information
        img1 = cv2.imread(os.path.join("task23", name, "im1.png"))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(os.path.join("task23", name, "im2.png"))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        data = np.load(os.path.join("task23", name, "data.npz"))
        pts1 = data['pts1'].astype(float)
        pts2 = data['pts2'].astype(float)
        K1 = data['K1']
        K2 = data['K2']
        shape = img1.shape
        Fund_mat.insert(0, name)
        #finding Fundamental matrix
        T1, T2, F, Fmat = find_fundamental_matrix(shape, pts1, pts2)
        Fund_mat.insert(1, F)
        Fund_mat.insert(2, Fmat)
        e1 , e2 = compute_epipoles(F)
        draw_epipolar(img1, img2, F, pts1, pts2, e1, e2, filename='xtrans')
        # you can check against this
        FCheck, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
        

        #######################################################################
        # TODO: Your code here                                                #
        #######################################################################
