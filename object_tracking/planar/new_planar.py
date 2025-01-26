import pyrealsense2 as rs
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor

# Path to the .bag file
bag_file_path = "test.bag"

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure to read from the bag file
config.enable_device_from_file(bag_file_path)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

print("[INFO] Starting playback from bag file...")
pipeline.start(config)

def segment_planes(depth_image, intrinsics, distance_threshold=0.05, min_plane_size=1000):
    """
    Segment planes in a depth image using RANSAC.
    """
    height, width = depth_image.shape
    points = []

    # Convert depth image to 3D points
    for y in range(height):
        for x in range(width):
            z = depth_image[y, x]
            if z > 0:  # Valid depth
                point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], z)
                points.append(point)

    points = np.array(points)
    planes = []
    depths = []

    while len(points) > min_plane_size:
        model = RANSACRegressor(residual_threshold=distance_threshold)
        X = points[:, :2]
        z = points[:, 2]
        model.fit(X, z)
        inliers = model.inlier_mask_
        if inliers.sum() < min_plane_size:
            break

        plane_mask = np.zeros_like(depth_image, dtype=np.uint8)
        avg_depth = np.mean(z[inliers])

        for idx, inlier in enumerate(inliers):
            if inlier:
                px, py = rs.rs2_project_point_to_pixel(intrinsics, points[idx])
                if 0 <= int(py) < height and 0 <= int(px) < width:
                    plane_mask[int(py), int(px)] = 255

        planes.append(plane_mask)
        depths.append(avg_depth)

        points = points[~inliers]

    return planes, depths

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            print("[INFO] End of bag file reached.")
            break
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = depth_image * depth_frame.get_units()

        profile = pipeline.get_active_profile()
        depth_stream = profile.get_stream(rs.stream.depth)
        intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

        planes, depths = segment_planes(depth_image, intrinsics)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        for idx, plane in enumerate(planes):
            mask = cv2.cvtColor(plane, cv2.COLOR_GRAY2BGR)
            depth_colormap = cv2.addWeighted(depth_colormap, 0.8, mask, 0.2, 0)
            cv2.putText(depth_colormap, f"Plane {idx + 1}: {depths[idx]:.2f}m", 
                        (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2)
        cv2.imshow("Planes", depth_colormap)

        if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
            break

finally:
    print("[INFO] Stopping playback and releasing resources...")
    pipeline.stop()
    cv2.destroyAllWindows()
