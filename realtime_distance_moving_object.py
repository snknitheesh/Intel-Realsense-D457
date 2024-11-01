import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# Constants for distance range in meters
MIN_DISTANCE = 0.5  # Minimum distance threshold (0.5 meters)
MAX_DISTANCE = 1.0  # Maximum distance threshold (1.0 meters)

# Initialize variables for frame differencing
previous_frame = None

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the RealSense camera to stream depth
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

# Allow the camera to warm up
time.sleep(1)

# Create an OpenCV window
cv2.namedWindow("Moving Objects (0.5m - 1m)", cv2.WINDOW_AUTOSIZE)

try:
    while True:
        # Wait for a frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Create a mask for objects within the specified range
        depth_mask = np.where((depth_image > MIN_DISTANCE * 1000) & (depth_image < MAX_DISTANCE * 1000), 255, 0).astype(np.uint8)

        # If there's a previous frame, calculate the difference to detect motion
        if previous_frame is not None:
            # Get the absolute difference between current and previous depth masks
            motion_mask = cv2.absdiff(previous_frame, depth_mask)

            # Threshold the motion mask to highlight moving areas
            _, moving_objects = cv2.threshold(motion_mask, 25, 255, cv2.THRESH_BINARY)

            # Create a color image for display purposes
            moving_display = cv2.applyColorMap(moving_objects, cv2.COLORMAP_JET)

            # Display the result
            cv2.imshow("Moving Objects (0.5m - 1m)", moving_display)

        # Update the previous frame
        previous_frame = depth_mask.copy()

        # Exit on pressing ESC
        if cv2.waitKey(1) == 27:
            break

finally:
    # Cleanup
    pipeline.stop()
    cv2.destroyAllWindows()