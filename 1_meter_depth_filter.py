import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a frame set
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame:
            continue
        
        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Create a binary mask where depths close to 1 meter are highlighted
        # Depth values are in millimeters, so 1000mm corresponds to ~1 meter
        depth_filtered = np.where((depth_image > 900) & (depth_image < 1100), 255, 0).astype(np.uint8)
        
        # Display the filtered depth image
        cv2.imshow("Filtered Depth at 1 Meter", depth_filtered)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()