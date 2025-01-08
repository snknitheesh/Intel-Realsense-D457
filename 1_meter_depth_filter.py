import pyrealsense2 as rs
import numpy as np
import cv2

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
        
        if not depth_frame:
            continue
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        original_color_image = color_image.copy()
        
        
        depth_distance_filtered = np.where((depth_image > 400) & (depth_image < 550), 255, 0).astype(np.uint8)
        depth_blurr_filtered = cv2.medianBlur(depth_distance_filtered, 15)

        original_depth_image = depth_blurr_filtered.copy()
        depth_bgr = cv2.cvtColor(depth_blurr_filtered, cv2.COLOR_GRAY2BGR)
        
        blurred_color = cv2.GaussianBlur(depth_blurr_filtered, (5, 5), 0)
        edges_color = cv2.Canny(blurred_color, 50, 100)
        contours_color, _ = cv2.findContours(edges_color, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_color:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                if cv2.isContourConvex(approx):
                    cv2.drawContours(depth_bgr, [approx], -1, (0, 255, 0), 3)
                    M = cv2.moments(contour)
                    if M['m00'] != 0: 
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])

                        cv2.circle(depth_bgr, (cx, cy), 5, (0, 0, 255), -1)
        

        cv2.imshow("Filtered Depth at 1 Meter", depth_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()