import pyrealsense2 as rs
import numpy as np
import cv2


def count_apples(image):

    # Convert image to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for red color (apples)
    lower_red1 = np.array([0, 100, 100])  # Lower range for red
    upper_red1 = np.array([10, 255, 255])  # Upper range for red
    lower_red2 = np.array([160, 100, 100])  # Another range for red
    upper_red2 = np.array([179, 255, 255])  # Another upper range for red

    # Threshold the HSV image to extract red colors
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Perform morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours of the apples
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    apple_count = 0

    for contour in contours:
        # Filter out small contours (noise)
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area for an apple
            apple_count += 1
            # Draw a bounding box around each apple
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return apple_count, image


# Configure color stream from Intel RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

try:
    # Start the pipeline
    pipeline.start(config)
    print("[INFO] Starting RealSense pipeline...")

    while True:
        # Wait for a color frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert the color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Count apples in the image
        apple_count, result_image = count_apples(color_image)

        # Display the result
        cv2.putText(result_image, f"Apples Count: {apple_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Apple Detection", result_image)

        # Exit on pressing ESC key
        if cv2.waitKey(1) == 27:
            break

finally:
    # Stop the pipeline and close the OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()
