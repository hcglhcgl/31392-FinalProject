import glob
import os


def stereo_calibration_images():
    return _read_stereo_image('../../Stereo_calibration_images/left*.png',
                              '../../Stereo_calibration_images/right*.png')


def stereo_occlusion_images():
    return _read_stereo_image('../../Stereo_conveyor_with_occlusions/left/*.png',
                              '../../Stereo_conveyor_with_occlusions/right/*.png')


def stereo_images():
    return _read_stereo_image('../../Stereo_conveyor_without_occlusions/left/*.png',
                              '../../Stereo_conveyor_without_occlusions/right/*.png')


def rectified():
    l, r = _read_stereo_image('../../rectified_images/left/*.png',
                             '../../rectified_images/right/*.png')
    return l, r


def _read_stereo_image(left_filter: str, right_filter: str):
    image_left = glob.glob(left_filter)
    image_right = glob.glob(right_filter)
    assert image_left
    assert image_right
    return (sorted(image_left, key=lambda i: int(os.path.splitext(os.path.basename(i))[0])),
            sorted(image_right, key=lambda i: int(os.path.splitext(os.path.basename(i))[0])))
