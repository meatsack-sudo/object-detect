import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection, YolosForObjectDetection, YolosImageProcessor
from PIL import Image
import numpy as np
from datetime import datetime

# Check if CUDA is available and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize the processor and model, and move the model to the GPU if available
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(device)

# model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny').to(device)
# processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

# Start capturing from the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

target_object = "person"

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image from BGR (OpenCV format) to RGB (PIL format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Process the frame and move inputs to the GPU
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # Post-process the outputs to get detection results
    target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Flag to check if target object is detected
    target_object_detected = False

    # Draw bounding boxes and labels on the frame
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = map(int, box)
        # Draw the bounding box
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        # Label the detection
        label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}"
        #print(label_text.partition(":")[0])

        if label_text.partition(":")[0] == target_object:
            target_object_detected = True
            print(f"{target_object.capitalize()} detected in frame at {datetime.now()}")

        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if target_object_detected:
        pass

    # Display the frame with detections
    cv2.imshow('Webcam Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
