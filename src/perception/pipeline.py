from skimage.metrics import structural_similarity
from src.calibration.image_parser import rectified
import cv2
import numpy as np

from src.perception.optical_flow import optical_flow_frames


def diff_areas(ref_img: np.ndarray, img: np.ndarray, blur=9) -> np.ndarray:
    # Convert images to grayscale
    grey_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur both images to make the SSIM different lower
    grey_ref = cv2.blur(grey_ref, (blur, blur))
    grey_img = cv2.blur(grey_img, (blur, blur))

    # obtain SSIM between frames
    (score, diff) = structural_similarity(grey_ref, grey_img, full=True)

    diff = (diff * 255).astype("uint8")
    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    mask = np.zeros_like(img)
    # mask = np.zeros(img.shape, np.uint8)
    for c in contours:
        area = cv2.contourArea(c)
        # taking into consideration differences with big areas.
        if area > 2000:
            x, y, w, h = cv2.boundingRect(c)
            mask = cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
            mask[y:y + h, x:x + w] = img[y:y + h, x:x + w]
    #kernel = np.ones((5, 5), np.uint8)
    #thresh = cv2.dilate(thresh, kernel, iterations=3)
    #mask = cv2.bitwise_and(mask, mask, mask=thresh)
    return mask


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


def feature_matching(first: np.ndarray, second: np.ndarray):
    first_grey = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second_grey = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(first_grey, None)
    kp2, des2 = orb.detectAndCompute(second_grey, None)
    matcher = cv2.BFMatcher()
    matches = matcher.match(des, des2)
    matches = sorted(matches, key=lambda x: x.distance)


def main():
    (left_imgs, right_imgs) = rectified()
    previous_img = cv2.imread(left_imgs[0])
    points = None
    for idx, img in enumerate(left_imgs):
        if idx == 0:
            continue
        frame = cv2.imread(img)
        moved_img = diff_areas(previous_img, crop_left(frame))
        points = optical_flow_frames(previous_img, moved_img, points)
        showing_img = moved_img.copy()
        for point in points:
            a, b = point.ravel()
            showing_img = cv2.circle(showing_img, (int(a), int(b)), 20, (100, 100, 100), 5)

        # TODO feature matching
        cv2.imshow(str(idx), showing_img)

        previous_img = moved_img
        cv2.waitKey(5000)


if __name__ == '__main__':
    main()
