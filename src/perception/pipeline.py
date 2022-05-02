from skimage.metrics import structural_similarity
from src.calibration.image_parser import rectified
import cv2
import numpy as np

def diff_areas(ref_img: np.ndarray, img: np.ndarray, blur = 15) -> np.ndarray:
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
    mask = np.zeros(img.shape, np.uint8)
    for c in contours:
        area = cv2.contourArea(c)
        # taking into consideration differences with big areas.
        if area > 2000:
            x, y, w, h = cv2.boundingRect(c)
            mask[y:y + h, x:x + w] = img[y:y + h, x:x + w]
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    mask = cv2.bitwise_and(mask, mask, mask=thresh)
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


def main():
    (left_imgs, right_imgs) = rectified()
    for idx, img in enumerate(left_imgs):
        if idx == 0:
            continue


if __name__ == '__main__':
    main()