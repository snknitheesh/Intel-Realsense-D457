import pyrealsense2 as rs
import numpy as np
import cv2

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

pipeline = rs.pipeline()
config = rs.config()

# Set the path to the bag file
bag_file = "kinisi.bag"  # Replace with your .bag file name or path
config.enable_device_from_file(bag_file)

# Dynamically enable streams based on the .bag file
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
available_streams = [s.stream_type() for s in pipeline_profile.get_streams()]

if rs.stream.color in available_streams:
    config.enable_stream(rs.stream.color)
else:
    print("Error: The .bag file does not contain a color stream.")
    exit(1)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Detection (RealSense)", color_image)

        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
