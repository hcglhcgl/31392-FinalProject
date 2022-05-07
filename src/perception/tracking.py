from src.calibration.image_parser import rectified, rectified_occlusion
import math
import cv2
import numpy as np

def background_preprocess(frame, ksize=5):
    frame = cv2.GaussianBlur(frame, (ksize, ksize), 2)
    return frame

def contour(static_img, dilate):
    kernel = np.ones((3, 3), np.uint8)
    static_img = background_preprocess(static_img)
    canny1 = cv2.Canny(static_img, 10, 150)
    mask = cv2.dilate(canny1, kernel, iterations=dilate)
    return mask


def crop_left(img):
    # draw filled rectangle in white on black background as mask
    mask = np.zeros_like(img)
    mask = cv2.rectangle(mask, (360, 230), (1273, 684), (255, 255, 255), -1)
    # apply mask to image
    result = cv2.bitwise_and(img, mask)
    return result


def track_movement_masks(static, frame_mask) -> np.ndarray:
    movement = cv2.bitwise_not(cv2.bitwise_or(static, cv2.bitwise_not(frame_mask)))
    # dilate movement so we connect edges from the object we find.
    kernel = np.ones((7, 7), np.uint8)
    return cv2.dilate(movement, kernel, iterations=4)


def moving_features(prev_frame, frame):
    def distance_points(x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)


    features = []
    feat1 = cv2.goodFeaturesToTrack(prev_frame, maxCorners=700, qualityLevel=0.0015, minDistance=6)
    if feat1 is None:
        return features
    feat2, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, frame, feat1, None)
    for i in range(len(feat1)):
        f10 = int(feat1[i][0][0])
        f11 = int(feat1[i][0][1])
        f20 = int(feat2[i][0][0])
        f21 = int(feat2[i][0][1])
        if distance_points(f10, f11, f20, f21) > 20:
            features.append(feat2[i])
    return features

def display_features(movement, frame, features):
    for feature in features:
        cv2.circle(frame, (int(feature[0][0]), int(feature[0][1])), 5, (0, 255, 0), -1)

    display = np.hstack((frame, cv2.cvtColor(movement, cv2.COLOR_GRAY2BGR)))
    cv2.imshow('show', display)
    cv2.waitKey(500)

def display_boxes(movement, frame, boxes):
    for box in boxes:
        x, y, w, h = cv2.boundingRect(box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    display = np.hstack((frame, cv2.cvtColor(movement, cv2.COLOR_GRAY2BGR)))
    cv2.imshow('show', display)
    cv2.waitKey(500)
    return frame

def find_objects(tracked_movement):
    cnts = cv2.findContours(tracked_movement, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts



def main(start, record):
    # check here how variables are initialized. images are objects and contains references to it. use copy so we make
    # sure we do not use the same when changing the variable.
    # basically we keep track of the first frame, current frame and previous frame.
    (left_imgs, right_imgs) = rectified_occlusion()
    previous_img = cv2.imread(left_imgs[0])
    left_imgs = left_imgs[start:-1]

    # generate a static backfround for the first 70 frames. No movement happens in this first frames.
    static_video = left_imgs[0:70]
    static_background = contour(previous_img, 4)
    for img in static_video:
        static_background = cv2.bitwise_or(static_background, contour(cv2.imread(img), 4))
        cv2.imshow('static', static_background)
        cv2.waitKey(500)



    # processing now the video.
    left_imgs = left_imgs[70+start: -1]
    for idx, img in enumerate(left_imgs):
        # skipping first tracking
        if idx == 0:
            frame = cv2.imread(img)
            if record:
                height, width, layers = frame.shape
                out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))
            previous_img = cv2.imread(img)
            mask = contour(previous_img, 1)
            previous_movement = track_movement_masks(static_background, mask)
            continue

        frame = cv2.imread(img)
        mask = contour(frame, 1)
        mask = crop_left(mask)
        tracked_movement = track_movement_masks(static_background, mask)
        boxes = find_objects(tracked_movement)
        display = display_boxes(tracked_movement, frame, boxes)
        out.write(frame)

        previous_movement = tracked_movement
    out.release()

if __name__ == '__main__':
    # you can pass here the frame number and it will start from that frame. Useful for debugging ...
    main(0, False)
