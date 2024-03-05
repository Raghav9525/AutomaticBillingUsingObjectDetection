
import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
import cvzone
from ultralytics import YOLO
import constant
import math

# Path to your trained YOLOv8 model weights
model_path = r".\models\best7.pt"  # Update with your model's path
model = YOLO(model_path)

classNames = constant.classNames

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0)

# Set desired size if necessary (may depend on your model's requirements)
desired_size = (640, 640)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    with torch.no_grad():
        detections = model(frame)[0]
        # conf_threshold = 0.5
        # nms_threshold = 0.12
        # detections = non_max_suppression(detections, conf_threshold, nms_threshold)

    for detection in detections:
        boxes = detection.boxes
        for box in boxes:
            print("hey")
            print(box.xyxy[0])
            x1, y1, x2, y2 = box.xyxy[0]
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Draw rectangle on the image array
            conf = math.ceil((box.conf[0] * 100)) / 100
            cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=9, rt=2, colorR=(255, 0, 0))
            cls = int(box.cls[0])
            CurrentClass = classNames[cls]
            if CurrentClass == "samosa" and conf > 0.9:
                cvzone.putTextRect(frame, f'{CurrentClass}', (x1, y1 - 10), scale=1, thickness=2, offset=0)
    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
