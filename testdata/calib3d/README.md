## TestData for fisheye >180 degrees intrinsics calibration

### Test code
* main.cpp 

#### Context
* OpenCV 3.4.3-4.2.0 Ubuntu 18.04, Cmake ~3.10
    
#### Deps
* opencv highgui, calib3d, imgproc

#### Usage
* ./test-bin --imgs-dir "./fisheye/51-50-00-43-f8-00/"
    
### Dataset images
* fisheye/51-50-00-43-f8-00/

### Result examples
* Fixed intrinsics calibration: `fix2.png fix.png`

* Original intrinsics calibration: `orig.png orig2.png`

### What to do?
* Check differnece of undistortion on: `fix.png` and `orig.png`
* Check quality of projected 3D points on: `fix2.png` `orig2.png`
* Compare reprojection errors of cv::fisheye::calibrate(): `stdout` 
