import numpy as np
from numpy.linalg import inv, norm
from find_jacobian import find_jacobian
from dcm_from_rpy import dcm_from_rpy
from rpy_from_dcm import rpy_from_dcm

def pose_estimate_nls(K, Twcg, Ipts, Wpts):
    """
    Estimate camera pose from 2D-3D correspondences via NLS.

    The function performs a nonlinear least squares optimization procedure 
    to determine the best estimate of the camera pose in the calibration
    target frame, given 2D-3D point correspondences.

    Parameters:
    -----------
    Twcg  - 4x4 homogenous pose matrix, initial guess for camera pose.
    Ipts  - 2xn array of cross-junction points (with subpixel accuracy).
    Wpts  - 3xn array of world points (one-to-one correspondence with Ipts).

    Returns:
    --------
    Twc  - 4x4 np.array, homogenous pose matrix, camera pose in target frame.
    """
    maxIters = 250                          # Set maximum iterations.

    tp = Ipts.shape[1]                      # Num points.
    ps = np.reshape(Ipts, (2*tp, 1), 'F')   # Stacked vector of observations.

    dY = np.zeros((2*tp, 1))                # Residuals.
    J  = np.zeros((2*tp, 6))                # Jacobian.

    #--- FILL ME IN ---
    iter = 0 
    # print(Wpts[:, 0])

    # Initialize Matrix 
    e = epose_from_hpose(Twcg)
    T = Twcg
    #Prepare K Matrix in homogenous form 
    K_h = np.hstack((K, [[0], [0], [0]]))           
    K_h = np.vstack((K_h, [0 ,0, 0, 1]))


    while iter <= maxIters:
        sum_JTJ = np.zeros((6,6))
        sum_JTE = np.zeros((6,1))
        for i in range(tp):
            Wpt = Wpts[:, i].reshape((3,1))
            Ipt = Ipts[:, i].reshape((2,1))


            J = find_jacobian(K, Twcg, Wpt)
            sum_JTJ += J.T @ J
            
            Wpt = np.vstack((Wpt, 1))
            Wpt.reshape((4,1))
                        
            E_x = K_h @ np.linalg.inv(T) @ Wpt
            E_x = (E_x/E_x[2])[:2]
            
            sum_JTE += J.T @ (Ipt - E_x)
            
        iter += 1   
        delta_x = (np.linalg.inv(sum_JTJ)) @ sum_JTE
        e += delta_x
        T = hpose_from_epose(e)

    Twc = T
    #------------------
    return Twc

#----- Functions Go Below -----

def epose_from_hpose(T):
    """Euler pose vector from homogeneous pose matrix."""
    E = np.zeros((6, 1))
    E[0:3] = np.reshape(T[0:3, 3], (3, 1))
    E[3:6] = rpy_from_dcm(T[0:3, 0:3])
  
    return E

def hpose_from_epose(E):
    """Homogeneous pose matrix from Euler pose vector."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1
  
    return T
