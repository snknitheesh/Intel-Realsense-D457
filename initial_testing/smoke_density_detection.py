# Smoke density detection
import pyrealsense2 as rs
import numpy as np
import cv2

# Thresholds for detecting smoke
SMOKE_THRESHOLD = 50  # Adjust based on sensitivity requirements

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Initialize previous frame for motion analysis
previous_frame = None

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert color image to grayscale for smoke detection
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Initialize previous frame if not done already
        if previous_frame is None:
            previous_frame = gray_image
            continue

        # Compute absolute difference between current and previous frame
        frame_diff = cv2.absdiff(previous_frame, gray_image)

        # Threshold the frame difference to detect significant changes
        _, smoke_mask = cv2.threshold(frame_diff, SMOKE_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Calculate smoke density (percentage of pixels in mask)
        smoke_density = (np.sum(smoke_mask) / (smoke_mask.size * 255)) * 100

        # Display the density on the output window
        display_image = color_image.copy()
        cv2.putText(display_image, f"Smoke Density: {smoke_density:.2f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show smoke detection output
        cv2.imshow("Smoke Detection", display_image)
        cv2.imshow("Smoke Mask", smoke_mask)

        # Update previous frame
        previous_frame = gray_image

        # Break the loop on ESC key press
        if cv2.waitKey(1) == 27:
            break

finally:
    # Stop streaming and close windows
    pipeline.stop()
    cv2.destroyAllWindows()
