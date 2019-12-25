import numpy as np
from imageio import imread
from mat4py import loadmat
from cross_junctions import cross_junctions
import cv2

# Load the boundary.
bpoly = np.array(loadmat("bounds.mat")["bpolyh1"])

# Load the world points.
Wpts = np.array(loadmat("world_pts.mat")["world_pts"])

# Load the example target image.
I = imread("example_target.png")
newImg = I.copy()
color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)

Ipts = cross_junctions(I, bpoly, Wpts)
saddle_point_list = Ipts.T
for i in range(len(saddle_point_list)):
    y = int(saddle_point_list[i][1])
    x = int(saddle_point_list[i][0])
    color_img.itemset((y, x, 0), 0)
    color_img.itemset((y, x, 1), 255)
cv2.imwrite("finalimage.png", color_img)


# You can plot the points to check!
print(Ipts)