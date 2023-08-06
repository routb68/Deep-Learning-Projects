<h2>Object Detection using YOLOv3</h2>

**Usage**

```
usage: yolo.py [-h] [--webcam WEBCAM] [--play_video PLAY_VIDEO]
               [--image IMAGE] [--video_path VIDEO_PATH]
               [--image_path IMAGE_PATH] [--verbose VERBOSE]

```

**To run the code**

1. Give proper path to the video/image file
2. Download the weight file "yolov3.weights" for YOLOv3 and set proper path in yolo.py
2. Run `bash run.sh`

**Files**

* run.sh -- to run the code
* yolo.py -- performs object detection 
* yolov3.cfg -- configuration file
* coco.names -- object classes available
