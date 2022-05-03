import numpy as np
import cv2


# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
def optical_flow_frames(first: np.ndarray, next: np.ndarray, p0=None):
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    old_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    if p0 is None:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    new_gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]

    return good_new.reshape(-1, 1, 2)