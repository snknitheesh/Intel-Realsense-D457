# Jetson RealSense Project

## Overview
This repository contains various Python scripts for working with Intel RealSense cameras on a Jetson platform. The scripts perform tasks such as shape detection, motion tracking, point cloud visualization, and depth filtering. Additionally, there are scripts for processing recorded `.bag` files to analyze object movement and distance.

## Files and Descriptions

### General Scripts
- **`commandpromptdisplay.py`** - Handles display of camera output on terminal.
- **`detect_shapes.py`** - Identifies and detects a rectangle shjapes object with its orientation.
- **`face_detection.py`** - Detects faces in real-time using the camera feed.
- **`motion_detection.py`** - Tracks motion within a given frame.
- **`openCVpointCloud.py`** - Uses OpenCV to visualize point cloud data.
- **`openCVviewer.py`** - Displays real-time camera feed using OpenCV.
- **`readBag.py`** - Reads `.bag` files recorded from the RealSense camera.

## Dependencies
To run these scripts, install the required dependencies:
```bash
pip install opencv-python numpy pyrealsense2
```

## Usage
1. **Live Camera Processing:**
   - Run `python detect_shapes.py` to identify shapes.
   - Run `python motion_detection.py` to track movement.
   - Run `python realtime_distance_moving_object.py` to track moving objects within a range.

2. **Processing `.bag` Files:**
   - Run `python readBag.py -i 'test.bag'` to read and analyze `.bag` files.

3. **Point Cloud Visualization:**
   - Run `openCVpointCloud.py` to view point cloud data.

## Future Improvements
- Implement deep learning models for enhanced object recognition.
- Optimize real-time filtering for better performance on Jetson devices.
- Enhance `.bag` file processing with additional filtering options.

## Author
This repository is developed for personal experiments and exploration with RealSense cameras and Jetson devices.

