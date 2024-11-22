import pyrealsense2 as rs
import numpy as np
import cv2

# Path to the .bag file
bag_file_path = "test.bag"

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure to read from the bag file
config.enable_device_from_file(bag_file_path)
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.depth, rs.format.z16, 30)
print("[INFO] Starting playback from bag file...")
pipeline.start(config)

def segment_planes(depth_image, distance_threshold=0.05, min_plane_size=1000):

    height, width = depth_image.shape
    points = []
    
    # Convert depth image to point cloud
    for y in range(height):
        for x in range(width):
            z = depth_image[y, x]
            if z > 0:
                points.append([x, y, z])
    points = np.array(points)

    planes = []
    depths = []

    while points.shape[0] > min_plane_size:
        # RANSAC to fit a plane
        plane_model, inliers = cv2.findHomography(
            srcPoints=points[:, :2],
            dstPoints=points[:, :2],
            method=cv2.RANSAC,
            ransacReprojThreshold=distance_threshold,
        )

        if len(inliers) < min_plane_size:
            break

        # Mask for the current plane
        mask = np.zeros(depth_image.shape, dtype=np.uint8)
        for inlier in inliers:
            x, y = int(points[inlier, 0]), int(points[inlier, 1])
            mask[y, x] = 255

        # Calculate average depth of the plane
        avg_depth = np.mean(depth_image[mask == 255])

        planes.append(mask)
        depths.append(avg_depth)

        # Remove inliers from points
        points = np.delete(points, inliers, axis=0)

    # Sort planes by depth
    sorted_indices = np.argsort(depths)
    planes = [planes[i] for i in sorted_indices]
    depths = [depths[i] for i in sorted_indices]

    return planes, depths

try:
    while True:
        # Get frames
        frames = pipeline.wait_for_frames()
        # color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            print("[INFO] End of bag file reached.")
            break

        # Convert images to numpy arrays
        # color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = depth_image * depth_frame.get_units()  # Convert to meters

        # Detect planar surfaces
        planes, depths = segment_planes(depth_image)

        # Display planes on the color image
        for idx, plane in enumerate(planes):
            mask = cv2.cvtColor(plane, cv2.COLOR_GRAY2BGR)
            # color_image = cv2.addWeighted(color_image, 0.8, mask, 0.2, 0)
            depth_image = cv2.addWeighted(depth_image, 0.8, mask, 0.2, 0)
            cv2.putText(depth_image, f"Plane {idx + 1}: {depths[idx]:.2f}m", 
                        (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2)

        # Show the output
        cv2.imshow("Planes", depth_image)

        if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
            break

finally:
    print("[INFO] Stopping playback and releasing resources...")
    pipeline.stop()
    cv2.destroyAllWindows()
