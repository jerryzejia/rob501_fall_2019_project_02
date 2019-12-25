import numpy as np
from numpy.linalg import inv, lstsq

# from scipy.ndimage import gaussian_filter

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'p' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array, subpixel location of saddle point in I (x, y coords).
    """
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

    parameters = lstsq(A, Z, rcond=None)[0]
    parameters = parameters.astype("float64")
    print(parameters)
    pt = np.dot(-inv(np.array([[2*parameters[0], parameters[1]], [parameters[1], 2*parameters[2]]])),  np.array([[parameters[3]], [parameters[4]]]))
    return pt
