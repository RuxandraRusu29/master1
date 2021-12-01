import cv2 as cv

import numpy as np
from matplotlib import pyplot as plt

img1 = cv.imread('a3p2a.png',0)
img2 = cv.imread('a3p2b.png',0)

# Initiate ORB detector, that it will be our detector object.
orb = cv.ORB_create()

# find the keypoints and compute the descriptors with ORB for images
kpts1, des1 = orb.detectAndCompute(img1, None)
kpts2, des2 = orb.detectAndCompute(img2, None)

# Create a BFMatcher object.
bf = cv.BFMatcher_create(cv.NORM_HAMMING)

# match descriptor
matches1to2 = bf.knnMatch(des1, des2, k=2)

# draw matches
good1to2 = []
for m, n in matches1to2:
    if m.distance < 0.6 * n.distance:
        good1to2.append(m)
        
print(good1to2)
src_pts = np.float32([kpts1[m.queryIdx].pt for m in good1to2]).reshape(-1, 2)
dst_pts = np.float32([kpts2[m.trainIdx].pt for m in good1to2]).reshape(-1, 2)
print(src_pts)
print(dst_pts)