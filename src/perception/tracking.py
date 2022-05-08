from src.calibration.image_parser import rectified, rectified_occlusion
import math
import cv2
import numpy as np
import glob
import os
import csv
import pickle
from sklearn.linear_model import LinearRegression


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


def display_boxes(movement, frame, x, y, w, h):
    if x == 0 and y == 0:
        pass
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    display = np.hstack((frame, cv2.cvtColor(movement, cv2.COLOR_GRAY2BGR)))
    cv2.imshow('show', display)
    cv2.waitKey(500)
    return frame

def get_xy(x, y, w, h):
    return (x + (w / 2)), (y + (h / 2))

def find_objects(tracked_movement):
    cnts = cv2.findContours(tracked_movement, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts

def denoiser(prev_x, current_x, prev_y, current_y):
    if current_x < 1030:
        return True

    result_y = np.abs(prev_y - current_y)
    result_x = np.abs(prev_x - current_x)
    if result_y > 8 or result_x > 8:
        return False
    else:
        return True

def filter_contours(cnts):
    filtered = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h > 50:
            filtered.append(c)
    return filtered


class Tracking:
    def __init__(self):
        self.x = self.y = self.h = self.w = 0

    def track(self, cnts):
        # first we obtain biggest contour
        if cnts is None or len(cnts) == 0:
            return 0, 0, 0, 0
        big_dim = 0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w + h >= big_dim:
                big = c
                big_dim = w + h

        (x, y, w, h) = self.__generate_box(big)
        if self.__is_close(big):
            self.__cache(x, y, w, h)
        else:
            cx, cy, cw, ch = cv2.boundingRect(big)
            self.__cache(cx, cy, cw, ch)
        return x, y, w, h

    def __generate_box(self, cnts):
        x, y, w, h = cv2.boundingRect(cnts)
        # if the original box is already X pixels in w then its to shrinking to much to predict
        if w < 20:
            return x, y, w, h
        wshift = max(self.w - w, 0)
        return x - wshift, y, w + wshift, h

    def __cache(self, x, y, w, h):
        """
        :param cnts: contour that is been caches in tracking.
        :return: None
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __is_close(self, cnts):
        """
        :param cnts: passing contour
        :return: returning boolean if previous contour is close to this one. We take into consideration width shrinking
        """
        x, y, w, h = cv2.boundingRect(cnts)
        wshift = max(self.w - w, 0)
        # if x of the contour has moved less than 30 pixels (counting the contour shrink)
        # checking also Y coordinate.
        return (self.x + wshift - x) < 30 and self.y - y < 50

def calculate_z(disparity):
    #f = 2.45mm
    #baseline = 120mm
    #senzoer width 9.6
    #1280
    if disparity == 0:
        disparity = 1
    f_in_pixels = 2.45/(9.6/1280)
    return f_in_pixels * (120 / disparity)

def crop_right(img):
    # draw filled rectangle in white on black background as mask
    mask = np.zeros_like(img)
    mask = cv2.rectangle(mask, (240, 230), (1186, 700), (255, 255, 255), -1)
    # apply mask to image
    result = cv2.bitwise_and(img, mask)
    return result

def main(start, record):
    # check here how variables are initialized. images are objects and contains references to it. use copy so we make
    # sure we do not use the same when changing the variable.
    # basically we keep track of the first frame, current frame and previous frame.
    (left_imgs, right_imgs) = rectified_occlusion()
    #left_imgs = images_left = glob.glob('C:/stuff/school/s2/perception for autonomous systems/exercises/final/rectified_w/l/*.png')
    #left_imgs = sorted(left_imgs, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    previous_img = cv2.imread(left_imgs[0])
    left_imgs = left_imgs[start:-1]

    # generate a static backfround for the first 70 frames. No movement happens in this first frames.
    static_video = left_imgs[0:70]
    static_background = contour(previous_img, 4)
    for img in static_video:
        static_background = cv2.bitwise_or(static_background, contour(cv2.imread(img), 4))
        cv2.imshow('static', static_background)
        cv2.waitKey(10)

    tracker = Tracking()
    # processing now the video.
    left_imgs = left_imgs[70 + start: -1]
    for idx, img in enumerate(left_imgs):
        # skipping first tracking
        if idx == 0:
            frame = cv2.imread(img)
            if record:
                height, width, layers = frame.shape
                out = cv2.VideoWriter('project_1.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))
            previous_img = cv2.imread(img)
            mask = contour(previous_img, 1)
            previous_movement = track_movement_masks(static_background, mask)
            continue

        frame = cv2.imread(img)
        mask = contour(frame, 1)
        mask = crop_left(mask)
        tracked_movement = track_movement_masks(static_background, mask)
        boxes = find_objects(tracked_movement)
        boxes = filter_contours(boxes)
        (x, y, w, h) = tracker.track(boxes)
        display = display_boxes(tracked_movement, frame, x, y, w, h)
        if record:
            out.write(frame)

        previous_movement = tracked_movement
    if record:
        out.release()


def kalman_main(start):
    ##### kalman #####
    F = np.array([[1, 1, 0.5 , 0, 0, 0, 0],  # x
                  [0, 1, 1, 0, 0, 0, 0],  # x'
                  [0, 0, 1, 0, 0, 0, 0],  # x''
                  [0, 0, 0, 1, 1, 0, 0],  # y
                  [0, 0, 0, 0, 1, 0, 0],  # y'
                  [0, 0, 0, 0, 0, 1, 1],  # z
                  [0, 0, 0, 0, 0, 0, 1]])  # z'

    # The observation matrix (2x6).
    H = np.array([[1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0], ])

    # The measurement uncertainty.
    R = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    I = np.identity(7)
    measurement_noise_cov = 0.7
    process_noise_cov = 0.001

    kalman = cv2.KalmanFilter(7, 3)
    kalman.transitionMatrix = np.array(F, np.float32)
    kalman.measurementMatrix = np.array(H, np.float32)
    kalman.measurementNoiseCov = np.array(R, np.float32) * measurement_noise_cov
    kalman.processNoiseCov = np.array(I, np.float32) * process_noise_cov

    ##### regression model loading #######
    lin = pickle.load(open('lin_model.sav', 'rb'))

    images_left = images_left = glob.glob(
        'C:/stuff/school/s2/perception for autonomous systems/exercises/final/rectified_w/l/*.png')
    images_right = glob.glob('C:/stuff/school/s2/perception for autonomous systems/exercises/final/rectified_w/r/*.png')
    images_left = sorted(images_left, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    images_right = sorted(images_right, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

    previous_img = cv2.imread(images_right[0])
    images_right = images_right[start:-1]

    previous_img_l = cv2.imread(images_right[0])
    images_left = images_left[start:-1]

    ### variables ####
    z = 300
    initial_state = [1150, 285, 550]
    previous_x = 0
    previous_y = 0
    denoiser_res = False
    idx = 0
    timer = 0
    #data_index = 0
    #data = []

    # generate a static backfround for the first 70 frames. No movement happens in this first frames.
    static_video = images_right[0:70]
    static_background = contour(previous_img, 4)
    for img in static_video:
        static_background = cv2.bitwise_or(static_background, contour(cv2.imread(img), 4))
        #cv2.imshow('static', static_background)
        #cv2.waitKey(10)

    static_video_l = images_left[0:70]
    static_background_l = contour(previous_img_l, 4)
    for img in static_video_l:
        static_background_l = cv2.bitwise_or(static_background_l, contour(cv2.imread(img), 4))

    # processing now the video.
    images_left = images_left[70 + start: -1]
    images_right = images_right[70 + start: -1]
    tracker = Tracking()

    for frame_l, frame_r in zip(images_left, images_right):
        # skipping first tracking
        if idx == 0:
            img_r = cv2.imread(frame_r)
            previous_img_r = cv2.imread(frame_r)
            mask_r = contour(previous_img_r, 1)
            previous_movement = track_movement_masks(static_background, mask_r)

            previous_img_l = cv2.imread(frame_l)
            mask_l = contour(previous_img_l, 1)
            previous_movement_l = track_movement_masks(static_background_l, mask_l)
            idx += 1
            continue

        img_r = cv2.imread(frame_r)
        img_l = cv2.imread(frame_l)
        original = img_r.copy()
        mask_r = contour(img_r, 1)
        mask_r = crop_right(mask_r)
        mask_l = contour(img_l, 1)
        mask_l = crop_left(mask_l)

        tracked_movement = track_movement_masks(static_background, mask_r)
        boxes = find_objects(tracked_movement)
        boxes = filter_contours(boxes)
        (old_x, old_y, w, h) = tracker.track(boxes)
        x, y = get_xy(old_x, old_y, w, h)
        #display = display_boxes(tracked_movement, img_r, old_x, old_y, w, h)

        tracked_movement_l = track_movement_masks(static_background_l, mask_l)
        boxes_l = find_objects(tracked_movement_l)
        boxes_l = filter_contours(boxes_l)
        (x_l, y_l, w_l, h_l) = tracker.track(boxes_l)
        x_l, y_l = get_xy(x_l, y_l, w_l, h_l)

        previous_movement = tracked_movement

        if boxes is not None:
            denoiser_res = denoiser(previous_x, x, previous_y, y)
            previous_x = x
            previous_y = y

            if x > 1130 and denoiser_res:
                kalman.__init__(7, 3)
                kalman.transitionMatrix = np.array(F, np.float32)
                kalman.measurementMatrix = np.array(H, np.float32)
                kalman.measurementNoiseCov = np.array(R, np.float32) * measurement_noise_cov
                kalman.processNoiseCov = np.array(I, np.float32) * process_noise_cov
                previous_x = x
                previous_y = y
                initial_state = [int(x), int(y), 550]
                #data_index = 0

            if x > 1090:
                timer = 0

        if boxes is not None and x != 0 and x < 1130 and denoiser_res:
            #data.append([data_index, x, y])
            #data_index += 1
            #timer += 1
            if boxes_l is not None and x_l != 0:
                disparity = x_l - x
                z = calculate_z(disparity)
            measurement = np.array(
                [[np.float32(x - initial_state[0])], [np.float32(y - initial_state[1])], [np.float32(z)]])
            kalman.correct(measurement)

            cv2.circle(original, (int(x), int(y)), 70, (0, 255, 0), 3)
            text = "Detection"
            cv2.putText(original, text, (int(x) - 80, int(y) - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)
            # if x == 0 and y == 0:
            #     pass
            # cv2.rectangle(original, (int(old_x), int(old_y)), (int(old_x) + int(w), int(old_y) + int(h)), (0, 255, 0), 2)

            ### Predict the next state
            prediction = kalman.predict()
            # cv2.circle(original, (int(prediction[0][0] + initial_state[0]), int(prediction[3][0]) + initial_state[1]),
            #            40, (255, 0, 0), 3)
            # text = "Kalman Prediction"
            # cv2.putText(original, text,
            #             (int(prediction[0][0] + initial_state[0]) - 50, int(prediction[3][0] + initial_state[1]) + 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            #             (255, 0, 0), 2, cv2.LINE_AA)

            ### Add text text
            text = "X: " + (int(x)).__str__()
            cv2.putText(original, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            text = "Y: " + (int(y)).__str__()
            cv2.putText(original, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            text = "Z: " + (int(z)).__str__()
            cv2.putText(original, text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            timer += 1
            cv2.imshow('Frame', original)
            cv2.waitKey(100)
        else:
            prediction = kalman.predict()
            text = "X: " + (int(prediction[0][0] + initial_state[0])).__str__()
            cv2.putText(original, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

            text = "Y: " + (int(prediction[3][0] + initial_state[1])).__str__()
            cv2.putText(original, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

            text = "Z: " + (int(prediction[6][0] + initial_state[2])).__str__()
            cv2.putText(original, text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.circle(original, (int(prediction[0][0] + initial_state[0]), int(prediction[3][0] + initial_state[1])),
                       70, (255, 0, 0), 2)

            text = "Kalman Prediction"
            cv2.putText(original, text, (int(prediction[0][0] + initial_state[0]) - 80, int(prediction[3][0] + initial_state[1]) + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0), 2, cv2.LINE_AA)

            ##### regression prediction #####
            time = np.array(timer-10).reshape(-1, 1)
            result = lin.predict(time)[0]
            result_x = int(result[0])
            result_y = int(result[1])
            if 530 < result_x < 1020:
                cv2.circle(original, (result_x, result_y), 70, (0, 211, 255), 2)

                text = "Linear Regression"
                cv2.putText(original, text, (result_x - 80, result_y - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 211, 255), 2, cv2.LINE_AA)
            timer += 1
            cv2.imshow('Frame', original)
            cv2.waitKey(100)

        idx += 1
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # you can pass here the frame number and it will start from that frame. Useful for debugging ...
    kalman_main(0)
