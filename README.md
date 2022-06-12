# Traffic sign detection and recognition using neural networks

This program takes dashcam footage and detects and recognises the traffic signs in it. 


Image/video -> Traffic sign detection network -> Traffic sign recognition network -> Output to user


The program contains 2 neural networks:

1. The first is for traffic sign detection 

2. The second is for traffic sign recognition

<br />


![Example 1](/examples/1.jpg)
![Example 2](/examples/2.jpg)
![Example 3](/examples/3.jpg)

## Prerequisites:

Go to https://drive.google.com/drive/folders/1cJl0CUJXfGHbd7LQWa1pcOIzKLrf2jdf?usp=sharing

Or  https://github.com/Alzaib/Traffic-Signs-Detection-Tensorflow-YOLOv3-YOLOv4

and download YOLO weight file, save as yolov3.weights and place in trafficSignDetection/yolo-coco-data folder

Need TensorFlow installed

<br />


## Main folder

image.py - Tests the complete network on images in trafficSignDetection/images folder - 1 example image is included

video.py - Tests the complete network on videos in trafficSignDetection/videos folder, results are saved in the videos/results folder - No example videos are included. Must place own video in directory and specify path in video.py


<br />

## Traffic Sign Detection folder

Uses YOLO3 for traffic sign detection. Database used was the GTDRB (German Traffic sign Detection Benchmark)


yolo-3-camera.py - Tests the YOLO network on real time camera footage

yolo-3-image.py - Tests the YOLO network on images in the images folder (1 included)

yolo-3-video.py - Tests the YOLO network on videos in the videos folder, results are save in the videos/results folder (None included)


<br />

## Traffic Sign Recognition folder

Databased used was the GTSRB (German Traffic Sign Recognition Benchmark)


road_sign2.1.h5 - The trained neural network 

tester.py - Tests the network on traffic signs in the images folder
- To change what image is tested change the path used in the cv2.imread function (5 images included)



