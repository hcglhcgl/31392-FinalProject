import numpy as np
import cv2


# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
def optical_flow_frames_corner(first: np.ndarray, next: np.ndarray, p0=None):
    # all this parameters where tuned, took a while to find something that would constantly find some good tracking points.

    # we are lowering the quality to find as many points from frame to frame. Every iteration we try to find the same point.
    # if no points are matched this process is repeated.
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.1,
                          minDistance=20,
                          blockSize=35)


    # 30 is for how many tries there is matching points.
    # 0.01 is epsilon or how much the point has to move between frames before is discarded. Static points are elimitated.
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(9, 9),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


    # Take first frame and find corners in it
    old_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    if p0 is None:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    new_gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    # Select good points, if no points matches then we return None
    try:
        good_new = p1[st == 1]
    except:
        return None
    return good_new.reshape(-1, 1, 2)

