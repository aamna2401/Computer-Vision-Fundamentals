import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt, pi, sin, cos, tan
from copy import deepcopy
import os

points1 = []
points2 = []
counter = 0 
H = [[1, 0, 0], [0, 1, 0]]

def read_img(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray


def retrieve_points(event, x, y, flags, params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        if counter < 4:
            points1.append([x, y])
            counter += 1
        else:
            points2.append([x, y])
    elif event == cv2.EVENT_LBUTTONUP:                   
        cv2.circle(img,(x,y), 4, (0,255,0), -1)
        cv2.imshow("image", img)



def get_points(func):
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", func)
    while True:
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF      
        
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)



def get_vanishing_line(points1, points2):
    points = deepcopy(points1)
    points.extend(points2)
    lines = []
    for i in range(0, len(points), 2):
        p1 = deepcopy(points[i])
        p2 = deepcopy(points[i+1])
        p1.append(1)
        p2.append(1)
        lines.append(np.cross(p1, p2))
    
    temp1 = np.cross(lines[0], lines[1])
    temp2 = np.cross(lines[2], lines[3])
    # Vanishing Points
    v1 = temp1/temp1[2]
    v2 = temp2/temp2[2]
    # vanishing line
    lv = np.cross(v1, v2)
    return lv


def get_Ha_matrix(lv):
    return np.vstack([H, [lv[0]/lv[2], lv[1]/lv[2], 1]])



def get_affine_img(Ha_mat, name):
    size = gray.shape
    affine_rect = cv2.warpPerspective(img, Ha_mat, (size[1], size[0]))
    cv2.imwrite(name+'_affCorrect.jpg', affine_rect)
    return affine_rect
    

perp_points1 = []
perp_points2 = []
perp_counter = 0


def retrieve_points_metric(event, x, y, flags, params):
    global perp_counter
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        if perp_counter < 3:
            perp_points1.append([x, y])
            perp_counter += 1
        else:
            perp_points2.append([x, y])
    elif event == cv2.EVENT_LBUTTONUP:                   
        cv2.circle(img,(x,y), 4, (0,255,0), -1)
        cv2.imshow("image", img)



def required_lines(perp_points1, perp_points2):
    points = deepcopy(perp_points1)
    points.extend(perp_points2)
    lines = []
    for i in range(0, len(points), 2):
        p1 = deepcopy(points[i])
        p2 = deepcopy(points[i+1])
        p1.append(1)
        p2.append(1)
        lines.append(np.cross(p1, p2))
    a = lines[0]/lines[0][2]
    b = lines[1]/lines[1][2]
    c = lines[2]/lines[2][2]
    d = lines[3]/lines[3][2]
    return a, b, c ,d



def symmetric_matrix(l1, m1, l2, m2):
    C = np.array([-l1[1]*m1[1], -l2[1]*m2[1]])
    A = np.array([[l1[0]*m1[0], l1[0]*m1[1] + l1[1]*m1[0]], [l2[0]*m2[0], l2[0]*m2[1] + l2[1]*m2[0]]])
    
    s = np.matmul(np.linalg.inv(A), C)
    
    # symmetric matrix
    s_whole = np.array([[s[0], s[1]], [s[1], 1]])
    return s_whole



def affine_hom_mat(sym_matrix):
    U, D, V = np.linalg.svd(sym_matrix)
    Dtemp = np.sqrt(D)
    Dfinal = np.array([[Dtemp[0], 0], [0, Dtemp[1]]])
    Am = np.matmul(np.matmul(U, Dfinal), V)
    return np.array([[Am[0][0], Am[0][1], 0], [Am[1][0], Am[1][1], 0], [0, 0, 1]])



def main():
    get_points(retrieve_points)
    lv = get_vanishing_line(points1, points2)
    Ha = get_Ha_matrix(lv)
    
    affine_rect = get_affine_img(Ha, name=os.path.join('result', img_name))
    img = affine_rect
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    get_points(retrieve_points_metric)
    
    a, b, c, d = required_lines(perp_points1, perp_points2)
    sym_mat = symmetric_matrix(a, b, c, d)
    
    H_aff = affine_hom_mat(sym_mat)
    inv_H_affine = np.float32(np.linalg.inv(H_aff))
    proj_corr = cv2.warpPerspective(affine_rect, inv_H_affine, size_new)
    cv2.imwrite(os.path.join('result', img_name+'_prCorrect.jpg'), proj_corr)


if __name__ == '__main__':
    os.makedirs('result', exist_ok=True)
    img_name = 'img7'
    img_path = os.path.join('images', img_name+'.jpg')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    size = gray.shape
    size_new = (size[1], size[0])
    main()
    