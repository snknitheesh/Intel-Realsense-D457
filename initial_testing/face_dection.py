# haarcascade_frontalface_default.xml  from https://github.com/opencv/opencv/tree/master/data/haarcascades
import pyrealsense2 as rs
import numpy as np
import cv2

# Path to the Haar Cascade file for face detection
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure RGB stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the camera pipeline
pipeline.start(config)

try:
    while True:
        # Wait for a frame and get the color frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert RealSense frame to a numpy array for OpenCV
        color_image = np.asanyarray(color_frame.get_data())

        # Convert to grayscale for face detection
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the image with detected faces
        cv2.imshow("Face Detection (RealSense)", color_image)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the RealSense pipeline and close windows
    pipeline.stop()
    cv2.destroyAllWindows()
