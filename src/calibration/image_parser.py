import glob


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
    return _read_stereo_image('../../rectified/left/*.png',
                              '../../rectified/right/*.png')


def _read_stereo_image(left_filter: str, right_filter: str):
    image_left = glob.glob(left_filter)
    right_filter = glob.glob(right_filter)
    assert image_left
    assert right_filter
    return sorted(image_left), sorted(right_filter)
