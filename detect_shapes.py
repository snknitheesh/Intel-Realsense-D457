import pyrealsense2 as rs
import numpy as np
import cv2

def pixel_to_point(depth, x, y, intrinsics):
    z = depth[y, x] / 1000.0  
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], z)
    return np.array(point)

def calculate_plane_normal(corners_3d):
    p1, p2, p3 = corners_3d[:3] 
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.array([0, 0, 0]) 
    normal = normal / norm 
    return normal

def calculate_orientation_angles(normal):
    a, b, c = normal
    pitch = np.arctan2(-a, np.sqrt(b**2 + c**2))  
    roll = np.arctan2(b, c)  
    yaw = np.arctan2(normal[1], normal[0])  
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def display_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6, color=(0, 255, 0), thickness=2, bg_color=(0, 0, 0)):
    x, y = position
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, -1)
    cv2.putText(image, text, position, font, scale, color, thickness)

def contour_detection(contours_depth):
    roll, pitch, yaw = 0, 0, 0
    for contour in contours_depth:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            cv2.drawContours(depth_bgr, [approx], -1, (0, 255, 0), 3)
            corners_3d = []
            for point in approx:
                x, y = point[0]
                corners_3d.append(pixel_to_point(depth_image, x, y, intrinsics))
            corners_3d = np.array(corners_3d)
            normal = calculate_plane_normal(corners_3d)
            roll, pitch, yaw = calculate_orientation_angles(normal)     
    return roll, pitch, yaw

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

        profile = pipeline.get_active_profile()
        intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        if not depth_frame:
            continue
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        original_color_image = color_image.copy()
        
        
        depth_distance_filtered = np.where((depth_image > 400) & (depth_image < 600), 255, 0).astype(np.uint8)
        depth_image_8bit = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_blurr_filtered = cv2.medianBlur(depth_distance_filtered, 15)

        original_depth_image = depth_blurr_filtered.copy()
    
        depth_bgr_extra = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_bgr_extra = cv2.cvtColor(depth_bgr_extra, cv2.COLOR_GRAY2BGR)
        depth_bgr = cv2.cvtColor(depth_blurr_filtered, cv2.COLOR_GRAY2BGR)
        
        blurred_color = cv2.GaussianBlur(depth_blurr_filtered, (5, 5), 0)
        edges_color = cv2.Canny(blurred_color, 50, 100)
        contours_depth, _ = cv2.findContours(edges_color, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        roll, pitch, yaw = contour_detection(contours_depth)

        display_text(depth_bgr, f"Roll: {roll:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        display_text(depth_bgr, f"Pitch: {pitch:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        display_text(depth_bgr, f"Yaw: {yaw:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        combined_display = np.hstack((depth_bgr, color_image))
        cv2.imshow("Object Detection", combined_display)

        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()



