import numpy as np
from scipy.ndimage.filters import *
from matplotlib.path import Path
# import cv2

def cross_junctions(I, bounds, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I       - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bounds  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts    - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of I. These should be floating-point values.
    """
    #--- FILL ME IN ---

    Ipts = np.zeros((2, 48))

    #------------------
    unit_square = np.array([[0.0, 100.0, 100.0, 0.0], [0.0, 0.0, 100.0, 100.0]])
    (H, A) = dlt_homography(bounds, unit_square)  # reuse code from project 1
    new_bounds = bounds.T.copy()
    # Calculate differences 
    x_up = np.abs(new_bounds[1][0] - new_bounds[0][0])
    x_down = np.abs(new_bounds[2][0] - new_bounds[3][0])
    y_right = np.abs(new_bounds[1][1] - new_bounds[2][1])
    y_left = np.abs(new_bounds[0][1] - new_bounds[3][1])
    
    reduction_constant = 0
    new_bounds[0] = [new_bounds[0][0] + 0.05* x_up, new_bounds[0][1] + 0.05 * y_left]
    new_bounds[1] = [new_bounds[1][0] - 0.05 * x_up, new_bounds[1][1] + 0.05 * y_right]
    new_bounds[2] = [new_bounds[2][0] - 0.05 * x_down, new_bounds[2][1] - 0.05 * y_right]
    new_bounds[3] = [new_bounds[3][0] + 0.05 * x_down, new_bounds[3][1] - 0.05 * y_left]
    # Create new boundaries to remove unwanted corners on the outside of the checker board
    path = Path(new_bounds)
    Ipts = harris_corner_detection(I, path)

    cluster_list = clustering(Ipts)
    saddle_point_list = []

    for ele in cluster_list:
        x = int(np.round(ele[0]))
        y = int(np.round(ele[1]))
        patch = I[y-25:y+25, x-25:x+25]
        s_point = saddle_point(patch).T[0]
        pt = []
        pt.append(s_point[0] + x - 25)
        pt.append(s_point[1] + y - 25)
        saddle_point_list.append(pt)

    #Computer Homography 
    print (saddle_point_list)
    saddle_point_list = sort_saddle_point(saddle_point_list, H)
    print(np.array(saddle_point_list).T)
    return np.array(saddle_point_list).T

def harris_corner_detection(I, path):
    I = I/255.
    corner_list = [] 
    dy, dx = np.gradient(I)
    Ixx = gaussian_filter(dx**2, sigma = 1)
    Ixy = gaussian_filter(np.multiply(dx, dy), sigma = 1)
    Iyy = gaussian_filter(dy**2, sigma = 1)
    k = 0.04
    for y in range(I.shape[0]):
        for x in range(I.shape[1]):
            if path.contains_point([x, y]):
                A = np.array([
                                [Ixx[y, x], Ixy[y, x]], 
                                [Ixy[y, x], Iyy[y, x]]
                                ])
                            
                det = np.linalg.det(A)
                trace = np.trace(A)
                r = det - k * (trace**2)
                corner_list.append((x, y, r))

    corner_list.sort(key=lambda x: x[2], reverse = True)
    return corner_list[500:]

def clustering(Ipts):
    clusters_list = []
    threshold = 26
    for Ipt in Ipts:
        min_dis = float("inf")
        cluster = -1
        for i in range(len(clusters_list)):
            distance = np.sqrt((clusters_list[i][0][0] - Ipt[0]) ** 2 + (clusters_list[i][0][1] - Ipt[1]) ** 2)
            if distance < min_dis:
                min_dis = distance
                cluster = i
        if min_dis > threshold:
            Ipt = list(Ipt[:2])
            clusters_list.append([Ipt])
        else:
            Ipt = list(Ipt[:2])
            clusters_list[cluster].append(Ipt)

    clusters_list.sort(key=lambda x: len(x), reverse = True) 
    clusters_list = clusters_list[:48]
    for i in range(len(clusters_list)):
        clusters_list[i] = [sum(x)/len(x) for x in zip(*clusters_list[i])]
    return clusters_list

def saddle_point(I):
    #--- FILL ME IN ---

    m, n = I.shape
    # I = gaussian_filter(I, sigma=0.1)
    A = np.zeros((m*n, 6))
    Z = np.zeros((m*n))
    i = 0
    for y in range(m):
        for x in range(n):
            X = np.zeros(6)
            X[0], X[1], X[2], X[3], X[4], X[5] = x**2, x*y, y**2, x, y, 1 
            A[i] = X
            Z[i] = I[y, x]
            i += 1 
    #------------------
    # Solve for parameter.
    #parameters = np.dot(np.linalg.pinv(A), Z)

    parameters = np.linalg.lstsq(A, Z, rcond=None)[0]
    parameters = parameters.astype("float64")
    pt = np.dot(-np.linalg.inv(np.array([[2*parameters[0], parameters[1]], [parameters[1], 2*parameters[2]]])),  np.array([[parameters[3]], [parameters[4]]]))
    return pt

def sort_saddle_point(saddle_points, H):
    pt_map = {}
    for pt in saddle_points:
        pt.append(1)
        mapped_pt = H @ pt 
        mapped_pt /= mapped_pt[2]
        mapped_pt = mapped_pt[:2]
        pt_map[tuple(mapped_pt)] = pt[:2]
    k = list(pt_map.keys())
    k.sort(key = lambda x: x[1])
    for i in range(6):  # for each y , sort along x
        k[8 * i : 8 * (i + 1)] = list(
            sorted(k[8 * i : 8 * (i + 1)], key=lambda x: x[0])
        )
    return [pt_map[x] for x in k]

def dlt_homography(I1pts, I2pts):
    # Setting up A 
    A = np.zeros((8,9))
    A_row_index = 0
    for i in range(0, len(I1pts[0])):
        x = I1pts[0][i]
        y = I1pts[1][i]

        u = I2pts[0][i]
        v = I2pts[1][i]

        A[A_row_index] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
        A_row_index += 1
        A[A_row_index] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
        A_row_index += 1

    # Solve for H
    U,S,Vh = np.linalg.svd(A)
    h = Vh[-1,:] / Vh[-1,-1]
    H = h.reshape(3,3)
    return H, A

