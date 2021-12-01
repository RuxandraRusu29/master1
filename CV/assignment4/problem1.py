import numpy as np
import scipy.linalg as la

def transform(pts):
    """Point conditioning: scale and shift points into [-1, 1] range
    as suggested in the lecture.
    
    Args:
        pts: [Nx2] numpy array of pixel point coordinates
    
    Returns:
        T: [3x3] numpy array, such that Tx normalises 
            2D points x (in homogeneous form).
    
    """
    assert pts.ndim == 2 and pts.shape[1] == 2

    m, n = pts.shape
    #s_list for calculating the point which has max squared root distance
    s_list = []
    mean_x = 0
    mean_y = 0
    #Calculating mean of x's and y's to form a matrix
    for i in range(m):
        s_list.append(pts[i][0]^2 + pts[i][1]^2)
        mean_x += pts[i][0]
        mean_y += pts[i][1]

    s_val = math.sqrt(max(s_list))/2
    mean_x = mean_x/m
    mean_y = mean_y/m

    #Create T using s, mean_x, mean_y. 
    T = np.array([[1/s_val, 0, -mean_x/s_val], [0, 1/s_val, -mean_y/s_val], [0,0,1]])
    

    #
    # Your code goes here
    #

    assert T.shape == (3, 3)
    return T


def transform_pts(pts, T):
    """Applies transformation T on 2D points.
    
    Args:
        pts: (Nx2) 2D point coordinates
        T: 3x3 transformation matrix
    
    Returns:
        pts_out: (Nx3) transformed points in homogeneous form
    
    """
    assert pts.ndim == 2 and pts.shape[1] == 2
    assert T.shape == (3, 3)


    #
    # Your code goes here
    #
    pts_h = np.empty((pts.shape[0], 3))

    #reshape points to homogeneous form
    for i in range(pts.shape[0]):
        pts_h[i][0] = pts[i][0]
        pts_h[i][1] = pts[i][1]
        pts_h[i][2] = 1
    
    #Apply transformation matrix
    pts_h = pts_h@T

    assert pts_h.shape == (pts.shape[0], 3)
    return pts_h

def create_A(pts1, pts2):
    """Create matrix A such that our problem will be Ax = 0,
    where x is a vectorised representation of the 
    fundamental matrix.
        
    Args:
        pts1 and pts2: Nx2 numpy arrays corresponding to 2D points 
    
    Returns:
        A: numpy array
    """
    assert pts1.shape == pts2.shape

    #
    # Your code goes here
    #

    t1 = transform(pts1)
    r1 = transform_pts(pts1, t1)

    t2 = transform(pts2)
    r2 = transform_pts(pts2, t2)

    #from the normalized points, construct A. 
    result = np.empty((pts1.shape[0], 9))
    for i in range(result.shape[0]):
        x = r1[i][0]
        y = r1[i][1]
        xp = r2[i][0]
        yp = r2[i][1]
        result[i] = [x*xp, y*xp, xp, x*yp, y*yp, yp, x, y, 1]
    

    return result

def enforce_rank2(F):
    """Enforce rank 2 of 3x3 matrix
    
    Args:
        F: 3x3 matrix
    
    Returns:
        F_out: 3x3 matrix with rank 2
    """
    assert F.shape == (3, 3)
    
    #
    # Your code goes here
    #
    #First perform a SVD. 
    u,s,vt = np.linalg.svd(F)

    #Set the last singular value to 0.
    s[2] = 0

    #Re-construct matrix using modified s vector. 
    F_final = (u*s)@vt
    assert F_final.shape == (3, 3)
    return F_final

def compute_F(A):
    """Computing the fundamental matrix from F
    by solving homogeneous least-squares problem
    Ax = 0, subject to ||x|| = 1
    
    Args:
        A: matrix A
    
    Returns:
        f: 3x3 matrix subject to rank-2 contraint
    """
    
    #
    # Your code goes here
    #

    #Perform SVD on A
    u,s,vt = np.linalg.svd(A)

    #Pick last column from columns of matrix V and reshape to 3x3 matrix. 
    f_init = vt[-1][:]
    F = f_init.reshape((3,3))

    #Enforce F's rank to 2. 
    F1 = enforce_rank2(F)

    F_final = F1
    
    assert F_final.shape == (3, 3)
    return F_final

def compute_residual(F, x1, x2):
    """Computes the residual g as defined in the assignment sheet.
    
    Args:
        F: fundamental matrix
        x1,x2: point correspondences
    
    Returns:
        float
    """
    g = 0
    m,n = x1.shape

    #Transforming to homogeneous form. 
    x1_h = np.ones((m, n+1))
    x2_h = np.ones((m, n+1))
    x1_h[:,:-1] = x1
    x2_h[:,:-1] = x2
    
    #Calculate the residual for each points. 
    for i in range(m):
        g += np.abs(x1_h[i]@ F @ x2_h[i].transpose() )
    
    g = g/m

    

    #
    # Your code goes here
    #
    return g

def denorm(F, T1, T2):
    """Denormalising matrix F using 
    transformations T1 and T2 which we used
    to normalise point coordinates x1 and x2,
    respectively.
    
    Returns:
        3x3 denormalised matrix F
    """

    #
    # Your code goes here
    #

    result = T1.transpose() * F * T2
    return result

def estimate_F(x1, x2, t_func):
    """Estimating fundamental matrix from pixel point
    coordinates x1 and x2 and normalisation specified 
    by function t_func (t_func returns 3x3 transformation 
    matrix).
    
    Args:
        x1, x2: 2D pixel coordinates of matching pairs
        t_func: normalising function (for example, transform)
    
    Returns:
        F: fundamental matrix
        res: residual g defined in the assignment
    """
    
    assert x1.shape[0] == x2.shape[0]

    #
    # Your code goes here
    #

    T1 = transform(x1)
    T2 = transform(x2)

    u1 = transform_pts(x1, T1)
    u2 = transform_pts(x2, T2)

    A = create_A(x1, x2)
    F = denorm(compute_F(A), T1, T2)

    res = compute_residual(F, x1, x2)


    return F, res


def line_y(xs, F, pts):
    """Compute corresponding y coordinates for 
    each x following the epipolar line for
    every point in pts.
    
    Args:
        xs: N-array of x coordinates
        F: fundamental matrix
        pts: (Mx3) array specifying pixel corrdinates
             in homogeneous form.
    
    Returns:
        MxN array containing y coordinates of epipolar lines.
    """
    N, M = xs.shape[0], pts.shape[0]
    assert F.shape == (3, 3)
    
    #
    # Your code goes here
    #
    ys = np.zeros((M,N))
    lines = pts @ F
    for i in range(M):
        x1 = lines[i][0]
        y1 = lines[i][1]
        for j in range(N):
            ys[i][j] = y1 * (xs[j]/x1)


    assert ys.shape == (M, N)
    return ys


#
# Bonus tasks
#

import math

def transform_v2(pts):
    """Point conditioning: scale and shift points into [-1, 1] range.
    
    Args:
        pts1 and pts2: Nx2 numpy arrays corresponding to 2D points
    
    Returns:
        T: numpy array, such that Tx conditions 2D (homogeneous) points x.
    
    """
    
    #
    # Your code goes here
    #

    N,_ = pts.shape
    x = pts[:][0].T
    y = pts[:][1].T

    xm = np.mean(x)
    ym = np.mean(y)
    xs = np.var(x)
    ys = np.var(y)

    T = np.array([[1/xs, 0, -xm/xs], [0,1/ys, -ym/ys], [0,0,1]])

    
    return T


"""Multiple-choice question"""
class MultiChoice(object):

    """ Which statements about fundamental matrix F estimation are true?

    1. We need at least 7 point correspondences to estimate matrix F.
    2. We need at least 8 point correspondences to estimate matrix F.
    3. More point correspondences will not improve accuracy of F as long as 
    the minimum number of points correspondences are provided.
    4. Fundamental matrix contains information about intrinsic camera parameters.
    5. One can recover the rotation and translation (up to scale) from the essential matrix 
    corresponding to the transform between the two views.
    6. The determinant of the fundamental matrix is always 1.
    7. Different normalisation schemes (e.g. transform, transform_v2) may have
    a significant effect on estimation of F. For example, epipoles can deviate.
    (Hint for 7): Try using corridor image pair.)

    Please, provide the indices of correct options in your answer.
    """

    def answer(self):
        return [2, 4, 5]


def compute_epipole(F, eps=1e-8):
    """Compute epipole for matrix F,
    such that Fe = 0.
    
    Args:
        F: fundamental matrix
    
    Returns:
        e: 2D vector of the epipole
    """
    assert F.shape == (3, 3)
    
    #
    # Your code goes here
    #
    y = -F[0][0]/F[1][1]
    return np.array([1,y,0])
 

def intrinsics_K(f=1.05, h=480, w=640):
    """Return 3x3 camera matrix.
    
    Args:
        f: focal length (same for x and y)
        h, w: image height and width
    
    Returns:
        3x3 camera matrix
    """

    #
    # Your code goes here
    #
    K = np.array([[1.05, 0, 320], [0,1.05, 240], [0,0,1]])

    return K

def compute_E(F):
    """Compute essential matrix from the provided
    fundamental matrix using the camera matrix (make 
    use of intrinsics_K).

    Args:
        F: 3x3 fundamental matrix

    Returns:
        E: 3x3 essential matrix
    """

    #
    # Your code goes here
    #
    K = intrinsics_K()

    E = K.T @ F @ K

    return E