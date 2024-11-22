import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Default bag file path
bag_file = "test.bag"

# Check if the bag file exists in the current directory
if not os.path.exists(bag_file):
    print(f"The file {bag_file} was not found in the current directory.")
    exit()

# Constants for depth filtering (in meters)
MIN_DISTANCE = 0.9  # Minimum distance threshold
MAX_DISTANCE = 1.1  # Maximum distance threshold

# Function to apply yellow-to-red colormap only on objects within a specified range
def apply_yellow_red_filter(depth_image, min_distance=MIN_DISTANCE, max_distance=MAX_DISTANCE):
    # Normalize the depth image within the specified range
    depth_normalized = np.clip((depth_image - min_distance * 1000) / ((max_distance - min_distance) * 1000), 0, 1)
    depth_normalized = (depth_normalized * 255).astype(np.uint8)
    
    # Create a mask to highlight areas within the desired distance range
    mask = np.where((depth_image >= min_distance * 1000) & (depth_image <= max_distance * 1000), 255, 0).astype(np.uint8)
    
    # Apply yellow-to-red colormap using OpenCV COLORMAP_HOT
    color_map = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_HOT)

    # Apply the mask to the color-mapped image, setting areas outside range to black
    filtered_image = cv2.bitwise_and(color_map, color_map, mask=mask)

    return filtered_image

try:
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file)

    # Enable depth stream (update with actual resolution and fps if known)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)  # Adjust frame rate and format as needed

    pipeline.start(config)

    cv2.namedWindow("Depth Stream - Yellow to Red Filter", cv2.WINDOW_AUTOSIZE)

    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            print("No depth frame detected.")
            continue

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply the yellow-to-red filter only on objects within 1 meter
        depth_colored = apply_yellow_red_filter(depth_image)

        # Display the filtered depth image
        cv2.imshow("Depth Stream - Yellow to Red Filter", depth_colored)

        if cv2.waitKey(1) == 27:  # Exit on ESC
            break

    cv2.destroyAllWindows()

finally:
    pipeline.stop()