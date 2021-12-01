import math
import numpy as np
from scipy import ndimage
from scipy import ndimage as ndi
from scipy import signal
import matplotlib.pyplot as plt 
import copy
from PIL import Image

def gauss2d(shape, sigma):

  """create gaussian filter
  """
  i = (shape[0] - 1) / 2
  j = (shape[1] - 1) / 2
  #create a 2D linspace to be the support for the filter
  y,x = np.ogrid[-i:i+1,-j:j+1]
  gaussian = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
  sum = gaussian.sum()
  #normalization of the filter
  if sum != 0:
    gaussian /= sum
  return gaussian

def createfilters():
  """
  Returns:
    fx, fy: filters as described in the problem assignment
  """
  sigma = 0.9

  #blur filter
  gx = gauss2d((3,1), sigma)
  #derivative filter
  dx = [-0.5, 0, 0.5]
  
  fx = gx*dx
 
  return np.array(fx), np.array(fx).T


def filterimage(I, fx, fy):
  """ Filter the image with the filters fx, fy.
  You may use the ndimage.convolve scipy-function.

  Args:
    I: a (H,W) numpy array storing image data
    fx, fy: filters

  Returns:
    Ix, Iy: images filtered by fx and fy respectively
  """
  
  #the images are filtered with convolution
  Ix=ndi.convolve(I,fx, output=np.float64, mode='reflect')
  Iy=ndi.convolve(I,fy, output=np.float64, mode='reflect')

  return Ix,Iy


def detectedges(Ix, Iy, thr):
  """ Detects edges by applying a threshold on the image gradient magnitude.

  Args:
    Ix, Iy: filtered images
    thr: the threshold value

  Returns:
    edges: (H,W) array that contains the magnitude of the image gradient at edges and 0 otherwise
  """

  edges = np.sqrt(Ix**2 + Iy**2)

  #each edge is checked in comparison with the threshold
  for i in range(0, edges.shape[0]):
    for j in range(0, edges.shape[1]):
        if edges[i][j] < thr:
          edges[i][j] = 0
      
  return edges

def nonmaxsupp(edges, Ix, Iy):
  """ Performs non-maximum suppression on an edge map.

  Args:
    edges: edge map containing the magnitude of the image gradient at edges and 0 otherwise
    Ix, Iy: filtered images

  Returns:
    nms_edges: edge map where non-maximum edges are suppressed
  """
  pi = np.pi
  angle = []
  
  #the angles are computed
  #in case the pixel from the x image is 0 we decide the angle b the y coordinate
  for i in range(Ix.shape[0]):
    angle.append([])
    for j in range(Ix.shape[1]):
        if Ix[i][j] == 0:
            if Iy[i][j] > 0:
                angle[i].append(90)
            elif Iy[i][j] < 0:
                angle[i].append(-90)
            else:
                angle[i].append(0)
        else:
            angle[i].append(np.rad2deg(np.arctan(Iy[i][j]/Ix[i][j])))
        
  nms_edges = copy.deepcopy(edges)
  
  np.pad(angle,1)
  
    
  for i in range(1, Ix.shape[0] - 1):
    for j in range(1, Ix.shape[1] - 1):
      # handle top-to-bottom edges: theta in [-90, -67.5] or (67.5, 90]
      if angle[i][j] > 67.5 or angle[i][j] <= -67.5:
        if edges[i][j] < edges[i][j]:
          nms_edges[i][j] = 0
        elif edges[i][j] < edges[i+1][j]:
          nms_edges[i][j] = 0
      # handle left-to-right edges: theta in (-22.5, 22.5]
      elif angle[i][j] <= 22.5 and angle[i][j] > -22.5:
        if edges[i][j] < edges[i][j-1]:
          nms_edges[i][j] = 0
        elif edges[i][j] < edges[i][j+1]:
          nms_edges[i][j] = 0
      # handle bottomleft-to-topright edges: theta in (22.5, 67.5]
      elif angle[i][j] <= 67.5 and angle[i][j] >  22.5:
        if edges[i][j] < edges[i-1][j-1]:
          nms_edges[i][j] = 0
        elif edges[i][j] < edges[i+1][j+1]:
            nms_edges[i][j]=0
      # handle topleft-to-bottomright edges: theta in [-67.5, -22.5]
      elif angle[i][j] >= -67.5 and angle[i][j] <= -22.5:
        if edges[i][j] < edges[i+1][j-1]:
          nms_edges[i][j] = 0
        if edges[i][j] < edges[i-1][j+1]:
          nms_edges[i][j] = 0

  return np.array(nms_edges)
  
if __name__ == '__main__':

    image = Image.open('Gura_Portitei_Scara_010.jpg')
    # summarize some details about the image
    print(image.format)
    print(image.mode)
    print(image.size)
    # show the image
    image.show()
