import pyrealsense2 as rs
import numpy as np
import cv2

MIN_DISTANCE = 0.5  
MAX_DISTANCE = 1.0  

previous_frame = None

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_mask = np.where((depth_image > MIN_DISTANCE * 1000) & (depth_image < MAX_DISTANCE * 1000), 255, 0).astype(np.uint8)

        if previous_frame is not None:
            motion_mask = cv2.absdiff(previous_frame, depth_mask)

            _, moving_objects = cv2.threshold(motion_mask, 25, 255, cv2.THRESH_BINARY)
            moving_display = cv2.applyColorMap(moving_objects, cv2.COLORMAP_JET)

            combined_display = np.hstack((color_image, moving_display))
            cv2.imshow("RGB and Moving Objects (0.5m - 1m)", combined_display)

        previous_frame = depth_mask.copy()

        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

