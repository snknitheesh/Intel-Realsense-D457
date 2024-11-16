import pyrealsense2 as rs
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# TensorRT setup
MODEL_PATH = "ssd_mobilenet_v2.trt"  # TensorRT model path
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Load TensorRT engine
def load_engine(model_path):
    with open(model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

print("[INFO] Loading TensorRT engine...")
engine = load_engine(MODEL_PATH)
context = engine.create_execution_context()

# Allocate device memory
def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})
    return inputs, outputs, bindings, stream

inputs, outputs, bindings, stream = allocate_buffers(engine)

# Inference function
def infer(image, context, inputs, outputs, bindings, stream):
    np.copyto(inputs[0]["host"], image.ravel())
    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
    stream.synchronize()
    return outputs[0]["host"].reshape(1, -1)

# Configure RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

try:
    print("[INFO] Starting RealSense pipeline...")
    pipeline.start(config)

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Get the color image
        color_image = np.asanyarray(color_frame.get_data())

        # Preprocess the image for TensorRT (resize, normalize)
        input_image = cv2.resize(color_image, (300, 300))  # Adjust size based on the model
        input_image = input_image.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))  # HWC to CHW

        # Perform inference
        detections = infer(input_image, context, inputs, outputs, bindings, stream)

        # Parse detections and draw bounding boxes
        h, w, _ = color_image.shape
        detection_threshold = 0.5  # Confidence threshold

        for detection in detections:
            confidence = detection[2]
            if confidence > detection_threshold:
                class_id = int(detection[1])
                xmin, ymin, xmax, ymax = (
                    int(detection[3] * w),
                    int(detection[4] * h),
                    int(detection[5] * w),
                    int(detection[6] * h),
                )
                # Draw bounding box
                label = f"Class {class_id} ({confidence:.2f})"
                cv2.rectangle(color_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(color_image, label, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the image
        cv2.imshow("Fruit Detection", color_image)

        # Break loop on ESC key
        if cv2.waitKey(1) == 27:
            break

finally:
    print("[INFO] Stopping RealSense pipeline...")
    pipeline.stop()
    cv2.destroyAllWindows()
