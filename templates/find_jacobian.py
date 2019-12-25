import numpy as np

def find_jacobian(K, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera pose optimization.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate, given a landmark point. The 
    projection model is the simple pinhole model.

    Parameters:
    -----------
    K    - 3x3 np.array, camera intrinsic calibration matrix.
    Twc  - 4x4 np.array, homogenous pose matrix, current guess for camera pose. 
    Wpt  - 3x1 world point on calibration target (one of n).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
    """
    #--- FILL ME IN ---

    #------------------
    J = np.zeros((2,6))
    P = Wpt
    R = Twc[0:3, 0:3]
    D = np.array(Twc[0:3, 3])
    
    #print(D)
    # All notation is from Szilisk Book
    #print(P)
    y1 = D -  P.T 
    y1 = y1.T
    y2 = np.matmul(R.T, y1) 
    z = y2[2]
    y3 = y2/z

    dxdy3 = K
    dy2dy1 = R.T
    #print(dy2dy1)

    dy3dy2 = (np.identity(3) * z - y2 * np.array([0,0,1]))/(z**2)

    # Find tx, ty, tz
    res = np.linalg.multi_dot([dxdy3, dy3dy2, dy2dy1])

    # Find dy2/dr, dy2/dp, dy2/dq

    # Copied from Given Function 
    rpy = np.zeros((3, 1))

    # Roll.
    rpy[0] = np.arctan2(R[2, 1], R[2, 2])

    # Pitch.
    sp = -R[2, 0]
    cp = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])

    if np.abs(cp) > 1e-15:
      rpy[1] = np.arctan2(sp, cp)
    else:
      # Gimbal lock...
      rpy[1] = np.pi/2
  
      if sp < 0:
        rpy[1] = -rpy[1]

    # Yaw.
    rpy[2] = np.arctan2(R[1, 0], R[0, 0])
    cr = np.cos(rpy[0]).item()
    sr = np.sin(rpy[0]).item()
    cp = np.cos(rpy[1]).item()
    sp = np.sin(rpy[1]).item()
    cy = np.cos(rpy[2]).item()
    sy = np.sin(rpy[2]).item()

    C3 = np.matrix([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
        ])
    C2 = np.matrix([
        [cp, 0,sp],
        [0, 1, 0],
        [-sp, 0, cp]
        ])

    C1 = np.matrix([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr,cr]
        ])

    dy2dy = (np.cross(np.array([0,0,1]).T, C3) @ C2 @ C1).T @ y1
    dy2dp = (C3 @ np.cross(np.array([0,1,0]).T, C2) @ C1).T @ y1
    dy2dr = (C3 @ C2 @ np.cross(np.array([1,0,0]).T, C1)).T @ y1

    row = dxdy3 @ dy3dy2 @ dy2dr
    pitch = dxdy3 @ dy3dy2 @ dy2dp
    yaw = dxdy3 @ dy3dy2 @ dy2dy

    J = np.hstack((res, -row, -pitch, -yaw))[0:2, :]
    return J
