# 31392-FinalProject

You are provided with: 
- Two datasets that contain: 
    - Raw stereo pair images (uncalibrated, unrectified) of objects on a conveyor. 
    - Raw stereo pair images (uncalibrated, unrectified) of objects on a conveyor with occlusion. Both datasets include multiple images of objects as they travel on the conveyor. The datasets contain images of 3 objects classes (cups, books and boxes). 
- Calibration pattern images for the used stereo camera camera.

Project Goals 
- Calibrate and rectify the stereo input. 
- Process the input images to detect objects on the conveyor and track them in 3D, even under occlusions. 
- Train a machine learning system that can classify unseen images into the 3 classes (cups, books and boxes) based either on 2D or 3D data. 
    - Use the web or/and capture your own images to create your training set. The image datasets provided with the project will constitute your testing set.git

## Method
### Calibration
To generate rectified images, run the "calibration_undistortion" script in src/calibration
### Detection
1. Take first frame as a reference.
2. For each next frame
3. Find different areas comparing it with the first frame.
4. Find features in stereo in the different areas.
5. TODO find a way to restart the features match
6. for the feature find XYZ
7. Kallman filter on XYZ.
8. If feature meassing then continue the point inside the convoy belt. (calculate velocity???)