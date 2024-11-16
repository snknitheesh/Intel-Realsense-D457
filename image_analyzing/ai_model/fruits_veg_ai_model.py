import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf

## warning we dont have a trained model 
#                              by fox

# Load the trained TensorFlow object detection model
MODEL_PATH = "saved_model"  # Path to your trained model
print("[INFO] Loading the AI model...")
detection_model = tf.saved_model.load(MODEL_PATH)

# Configure the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
print("[INFO] RealSense camera is ready!")

# Load category labels (label map file)
LABEL_MAP = {
    1: "Apple",
    2: "Banana",
    3: "Tomato",
    4: "Carrot",
    # Add more 
}

# Function to perform object detection
def detect_objects(image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]  # Expand dimensions for batch processing

    detections = detection_model(input_tensor)
    return detections

try:
    while True:
        # Wait for a frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Convert the image to RGB (TensorFlow requires RGB format)
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Run object detection
        detections = detect_objects(rgb_image)

        # Extract detection information
        detection_boxes = detections['detection_boxes'][0].numpy()
        detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
        detection_scores = detections['detection_scores'][0].numpy()

        # Draw detection results on the image
        for i in range(len(detection_boxes)):
            if detection_scores[i] > 0.5:  # Confidence threshold
                box = detection_boxes[i]
                class_id = detection_classes[i]
                label = LABEL_MAP.get(class_id, "Unknown")

                # Convert normalized box coordinates to image coordinates
                height, width, _ = color_image.shape
                ymin, xmin, ymax, xmax = box
                (left, top, right, bottom) = (int(xmin * width), int(ymin * height),
                                              int(xmax * width), int(ymax * height))

                # Draw bounding box and label
                cv2.rectangle(color_image, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(color_image, f"{label} ({detection_scores[i]:.2f})",
                            (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the output
        cv2.imshow("Fruit and Vegetable Detection", color_image)

        # Exit on pressing ESC
        if cv2.waitKey(1) == 27:
            break

finally:
    # Stop the camera pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
