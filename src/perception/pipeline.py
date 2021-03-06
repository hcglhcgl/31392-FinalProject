import numpy as np
from skimage.metrics import structural_similarity
import cv2
import sys
sys.path.append(
    r'C:\Users\krogh\Documents\DTU\Perception\31392-FinalProject\src')
# flfaer
from perception.optical_flow import optical_flow_frames_corner
from calibration.image_parser import rectified

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
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray, new_gray, p0, None, **lk_params)

    # Select good points, if no points matches then we return None
    try:
        good_new = p1[st == 1]
    except:
        return None
    return good_new.reshape(-1, 1, 2)


def treat_image(img: np.ndarray) -> np.ndarray:
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_cof = 9
    blur = cv2.blur(grey_img, (blur_cof, blur_cof))

    return blur


def diff_areas(ref_img: np.ndarray, img: np.ndarray, blur=13) -> (np.ndarray, bool, np.ndarray):
    # Convert images to grayscale
    grey_ref = treat_image(ref_img)
    grey_img = treat_image(img)

    # obtain SSIM between frames
    (score, diff) = structural_similarity(
        grey_ref, grey_img, full=True, win_size=21)

    diff = (diff * 255).astype("uint8")
    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(
        diff, 150, 250, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    mask = np.zeros_like(img)
    # mask = np.zeros(img.shape, np.uint8)
    masked = False
    nr_areas = 0
    for c in contours:
        area = cv2.contourArea(c)
        # taking into consideration differences with big areas.
        if area > 2000:
            nr_areas += 1
            masked = True
            x, y, w, h = cv2.boundingRect(c)
            mask = cv2.rectangle(
                mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
            mask[y:y + h, x:x + w] = img[y:y + h, x:x + w]
            middle_x = (x + x + w) / 2
            middle_y = (y + y + h) / 2
    if nr_areas >= 4:
        return np.zeros_like(img), False, thresh, middle_x, middle_y
    return mask, masked, thresh, middle_x, middle_y


def crop_right(img):
    # draw filled rectangle in white on black background as mask
    mask = np.zeros_like(img)
    mask = cv2.rectangle(mask, (240, 300), (1186, 700), (255, 255, 255), -1)
    # apply mask to image
    result = cv2.bitwise_and(img, mask)
    return result


def crop_left(img):
    # draw filled rectangle in white on black background as mask
    mask = np.zeros_like(img)
    mask = cv2.rectangle(mask, (360, 230), (1273, 684), (255, 255, 255), -1)
    # apply mask to image
    result = cv2.bitwise_and(img, mask)
    return result


def main(start):
    # check here how variables are initialized. images are objects and contains references to it. use copy so we make
    # sure we do not use the same when changing the variable.
    # basically we keep track of the first frame, current frame and previous frame.
    left_imgs, right_imgs = rectified()
    previous_img = cv2.imread(left_imgs[0])
    left_imgs = left_imgs[start:-1]
    first_img = previous_img.copy()
    points = None
    for idx, img in enumerate(left_imgs):
        if idx == 0:
            continue
        frame = cv2.imread(img)
        moved_img, masked, thresh = diff_areas(first_img, frame)
        moved_img = crop_left(moved_img)
        if masked is False:
            points = None
        else:
            points = optical_flow_frames_corner(
                previous_img, moved_img, points)

            showing_img = moved_img.copy()
            # optical flow can find no points. so we need to check for nulls before.
            if points is not None:
                for point in points:
                    # This are the points that follows the objects. Might get some outliers in this case, but they are pretty stable
                    a, b = point.ravel()
                    cv2.circle(moved_img, (int(a), int(b)), 20, (0, 0, 255), 5)

        # creating a display for visualizing the diff_area function and the tracking. We do not need this in the future.
        # frame is the original frame, and moved_img is the frame cropped and showing the points of optical flow.
        display = np.hstack(
            (moved_img, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)))

        cv2.destroyAllWindows()
        cv2.namedWindow(str(idx))
        cv2.moveWindow(str(idx), 40, 30)  # Move it to (40,30)
        displayS = cv2.resize(display, (960, 540))
        cv2.imshow(str(idx), displayS)
        previous_img = frame
        cv2.waitKey(20)


if __name__ == '__main__':
    # you can pass here the frame number and it will start from that frame. Useful for debugging ...
    main(0)
