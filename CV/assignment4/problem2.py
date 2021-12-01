import numpy as np

def cost_ssd(patch1, patch2):
    """Compute the Sum of Squared Pixel Differences (SSD):

    Args:
        patch1: input patch 1 as (m, m, 1) numpy array
        patch2: input patch 2 as (m, m, 1) numpy array

    Returns:
        cost_ssd: the calcuated SSD cost as a floating point value
    """
    # just subtract each other and squre it and calculate the sum
    cost_ssd = ((patch1-patch2)**2).sum()
    assert np.isscalar(cost_ssd)
    return cost_ssd


def cost_nc(patch1, patch2):
    """Compute the normalized correlation cost (NC):

    Args:
        patch1: input patch 1 as (m, m, 1) numpy array
        patch2: input patch 2 as (m, m, 1) numpy array

    Returns:
        cost_nc: the calcuated NC cost as a floating point value
    """
    m = patch1.shape[0]

    # make it to the shape m**2,1 so that easily calculate the process
    repatch1 = np.reshape(patch1, (m**2,1))
    repatch2 = np.reshape(patch2, (m**2,1))
    # Get the mean for each vector
    w1_mean = np.mean(repatch1)
    w2_mean = np.mean(repatch2)
    # Do the caculation as noted in the hw document
    cost_nc = np.dot(np.transpose(repatch1-w1_mean),repatch2-w2_mean)[0][0]
    cost_nc /= np.linalg.norm(repatch1-w1_mean)*np.linalg.norm(repatch2-w2_mean)

    assert np.isscalar(cost_nc)
    return cost_nc


def cost_function(patch1, patch2, alpha):
    """Compute the cost between two input window patches given the disparity:

    Args:
        patch1: input patch 1 as (m, m) numpy array
        patch2: input patch 2 as (m, m) numpy array
        input_disparity: input disparity as an integer value
        alpha: the weighting parameter for the cost function
    Returns:
        cost_val: the calculated cost value as a floating point value
    """

    assert patch1.shape == patch2.shape
    m = patch1.shape[0]

    # Using ssd and nc, caculate the cost from the document
    cost_val = cost_ssd(patch1, patch2)/(m**2) + alpha*cost_nc(patch1, patch2)

    assert np.isscalar(cost_val)
    return cost_val


def pad_image(input_img, window_size, padding_mode='symmetric'):
    """Output the padded image

    Args:
        input_img: an input image as a numpy array
        window_size: the window size as a scalar value, odd number
        padding_mode: the type of padding scheme, among 'symmetric', 'reflect', or 'constant'

    Returns:
        padded_img: padded image as a numpy array of the same type as image
    """
    assert np.isscalar(window_size)
    assert window_size % 2 == 1

    # Padding width must be window_size-1 and divided by 2. So that we can check every pixels
    pad_width = int((window_size-1)/2)
    # For each padding_mode, pad differently

    # But in result, I chose symmetric cause it seems to have smallest aepe
    if padding_mode == 'symmetric':
        padded_img = np.pad(input_img, pad_width, padding_mode)
    elif padding_mode == 'reflect':
        padded_img = np.pad(input_img, pad_width, padding_mode)
    elif padding_mode == 'constant':
        padded_img = np.pad(input_img, pad_width, padding_mode)

    return padded_img


def compute_disparity(padded_img_l, padded_img_r, max_disp, window_size, alpha):
    """Compute the disparity map by using the window-based matching:

    Args:
        padded_img_l: The padded left-view input image as 2-dimensional (H,W) numpy array
        padded_img_r: The padded right-view input image as 2-dimensional (H,W) numpy array
        max_disp: the maximum disparity as a search range
        window_size: the patch size for window-based matching, odd number
        alpha: the weighting parameter for the cost function
    Returns:
        disparity: numpy array (H,W) of the same type as image
    """

    assert padded_img_l.ndim == 2
    assert padded_img_r.ndim == 2
    assert padded_img_l.shape == padded_img_r.shape
    assert max_disp > 0
    assert window_size % 2 == 1

    # Each are original size of the Image
    origin_m = padded_img_l.shape[0]-window_size+1
    origin_n = padded_img_l.shape[1]-window_size+1

    # disparity should have same size with original image
    disparity = np.zeros((origin_m, origin_n))

    # For origin_m and origin_n times, need to be checked
    for i in range(origin_m):
        for j in range(origin_n):
            # Crop the image with window size
            patch1 = padded_img_l[i:i+window_size, j:j+window_size]
            # Make min_cost and min_disp have no value at the first time
            min_cost = None
            min_disp = None
            # Need to check left and right side as much as max_disp
            for k in range(-max_disp, max_disp+1):
                # when at the edge, cannot be checked
                if j-k<0:
                    continue
                if j-k+window_size> padded_img_r.shape[1]:
                    continue
                # Crop the right image to window size
                patch2 = padded_img_r[i:i+window_size, j-k:j+window_size-k]
                # Check the cost, and keep track the cost and disparity at the time
                cost = cost_function(patch1, patch2, alpha)
                if min_cost == None:
                    min_cost = cost
                    min_disp = k
                elif cost < min_cost:
                    min_cost = cost
                    min_disp = k
            # Put disparity at disparity map when it has the lowest cost
            disparity[i,j] = min_disp

    assert disparity.ndim == 2
    return disparity

def compute_aepe(disparity_gt, disparity_res):
    """Compute the average end-point error of the estimated disparity map:

    Args:
        disparity_gt: the ground truth of disparity map as (H, W) numpy array
        disparity_res: the estimated disparity map as (H, W) numpy array

    Returns:
        aepe: the average end-point error as a floating point value
    """
    assert disparity_gt.ndim == 2
    assert disparity_res.ndim == 2
    assert disparity_gt.shape == disparity_res.shape

    # Caculate the size to divide
    size = disparity_gt.shape[0]*disparity_gt.shape[1]
    # Get the aepe as the equation in the document
    aepe = np.sum(np.absolute(disparity_gt-disparity_res))/size

    assert np.isscalar(aepe)
    return aepe

def optimal_alpha():
    """Return alpha that leads to the smallest EPE
    (w.r.t. other values)"""

    # When I checked all of alphas, -0.01 was the best
    alpha = -0.01
    # np.random.choice([-0.06, -0.01, 0.04, 0.1])
    return alpha


"""
This is a multiple-choice question
"""
class WindowBasedDisparityMatching(object):

    def answer(self):
        """Complete the following sentence by choosing the most appropriate answer
        and return the value as a tuple.
        (Element with index 0 corresponds to Q1, with index 1 to Q2 and so on.)

        Q1. [?] is better for estimating disparity values on sharp objects and object boundaries.
          1: Using a smaller window size (e.g., 3x3)
          2: Using a bigger window size (e.g., 11x11)

        Q2. [?] is good for estimating disparity values on locally non-textured area.
          1: Using a smaller window size (e.g., 3x3)
          2: Using a bigger window size (e.g., 11x11)

        Q3. When using a [?] padding scheme, the artifacts on the right border of the estimated disparity map become the worst.
          1: constant
          2: reflect
          3: symmetric

        Q4. The inaccurate disparity estimation on the left image border happens due to [?].
          1: the inappropriate padding scheme
          2: the absence of corresponding pixels
          3: the limitations of the fixed window size
          4: the lack of global information

        """

        return (1, 2, 1, 2)
