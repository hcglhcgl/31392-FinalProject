import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.calibration.image_parser import stereo_calibration_images, stereo_images, stereo_occlusion_images


class ImageRectifier:
    @staticmethod
    def camera_mtx_from_images(image, nb_horizontal, nb_vertical):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((nb_horizontal * nb_vertical, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nb_vertical, 0:nb_horizontal].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        for fname in image:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Implement findChessboardCorners here
            ret, corners = cv2.findChessboardCorners(gray, (nb_vertical, nb_horizontal))

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints

    @staticmethod
    def calibrate_stereo_camera():
        images_left, images_right = stereo_calibration_images()
        rret, rmtx, rdist, rrvecs, rtvecs, robjpoints, rimgpoints = ImageRectifier.camera_mtx_from_images(images_right,
                                                                                                          9, 6)
        lret, lmtx, ldist, lrvecs, ltvecs, lobjpoints, limgpoints = ImageRectifier.camera_mtx_from_images(images_left,
                                                                                                          9, 6)

        return lret, lmtx, ldist, lrvecs, ltvecs, lobjpoints, limgpoints, \
               rret, rmtx, rdist, rrvecs, rtvecs, robjpoints, rimgpoints

    @staticmethod
    def undistor_images(images_path, mtx, dist):
        new_images = []
        for image_path in images_path:
            image = cv2.imread(image_path)
            h, w, _ = image.shape
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0)
            dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
            new_images.append(dst)
        return new_images

    @staticmethod
    def rectify_conveyor_images(occlusion=False):
        if occlusion:
            left_images, right_images = stereo_occlusion_images()
        else:
            left_images, right_images = stereo_images()

        _, lmtx, ldist, _, _, lobj, limg, _, rmtx, rdist, _, _, robj, rimg = ImageRectifier.calibrate_stereo_camera()

        h, w, _ = cv2.imread(left_images[0]).shape
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

        ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(lobj, limg, rimg,
                                                                      lmtx, ldist,
                                                                      rmtx, rdist, (w, h),
                                                                      criteria=criteria, flags=stereocalibration_flags)

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(lmtx, ldist, rmtx, rdist, (w, h), R, T)
        map1x, map1y = cv2.initUndistortRectifyMap(lmtx, ldist, R1, lmtx, (w, h), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(rmtx, rdist, R1, rmtx, (w, h), cv2.CV_32FC1)
        inter = cv2.INTER_LANCZOS4

        rectified_left = []
        rectified_right = []
        for img in left_images:
            i = cv2.imread(img)
            i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            rectified_left.append(cv2.remap(i, map1x, map1y, inter))
        for img in right_images:
            i = cv2.imread(img)
            i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            rectified_right.append(cv2.remap(i, map2x, map2y, inter))
        return rectified_left, rectified_right


def main():
    '''

    :return:
    '''
    left_convoy, right_convoy = ImageRectifier.rectify_conveyor_images()
    i = 0
    for img in left_convoy:
        plt.imshow(img)
        plt.imsave('../../rectified/left/' + str(i+10000) + '.png', img)
        i = i + 1

    i = 0
    for img in right_convoy:
        plt.imshow(img)
        plt.imsave('../../rectified/right/' + str(i+10000) + '.png', img)
        i = i + 1


if __name__ == '__main__':
    main()
