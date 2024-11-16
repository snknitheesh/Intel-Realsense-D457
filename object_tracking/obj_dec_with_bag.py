import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf

# Path to the .bag file
foxop = "20241112_223036.bag"

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Configure the pipeline to stream from the bag file
config.enable_device_from_file(foxop)  #bag_file_path

# Enable color stream
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

print("[INFO] Starting playback from bag file...")
pipeline.start(config)
print("[INFO] Bag file loaded and ready.")


print("[INFO] Loading TensorFlow model...")
PATH_TO_CKPT = "frozen_inference_graph.pb"

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)

# Input 
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Output
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

print("[INFO] Model loaded successfully.")
colors_hash = {}

try:
    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            print("[INFO] End of bag file reached.")
            break

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        scaled_size = (color_frame.width, color_frame.height)

        # Expand image dimensions to [1, None, None, 3]
        image_expanded = np.expand_dims(color_image, axis=0)

        # Perform object detection
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                 feed_dict={image_tensor: image_expanded})

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        for idx in range(int(num)):
            class_ = classes[idx]
            score = scores[idx]
            box = boxes[idx]

            if class_ not in colors_hash:
                colors_hash[class_] = tuple(np.random.choice(range(256), size=3))

            if score > 0.6:
                left = int(box[1] * color_frame.width)
                top = int(box[0] * color_frame.height)
                right = int(box[3] * color_frame.width)
                bottom = int(box[2] * color_frame.height)

                p1 = (left, top)
                p2 = (right, bottom)
                # Draw bounding box
                r, g, b = colors_hash[class_]
                cv2.rectangle(color_image, p1, p2, (int(r), int(g), int(b)), 2, 1)

        # Display the results
        cv2.namedWindow('RealSense - Bag File', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense - Bag File', color_image)

        if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
            break

finally:
    print("[INFO] Stopping playback and releasing resources...")
    pipeline.stop()
    cv2.destroyAllWindows()
