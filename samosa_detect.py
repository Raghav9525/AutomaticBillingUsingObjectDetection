import math

import cv2
# import torch
import cvzone
import numpy as np
# from models.experimental import attempt_load  # Adjust based on your directory structure
# from utils.general import non_max_suppression
from PIL import Image
from ultralytics import YOLO

from ultralytics import YOLO

# Load the YOLO model
model_path = "./models/best5.pt"  # Adjust with your model's path

# Load the image using OpenCV
image_path = "samosa.jpg"
image = Image.open(image_path)
img = cv2.imread(image_path)  # OpenCV loads images in BGR format


model = YOLO(model_path)

detections = model.predict(image_path)[0]
print(detections)

for detection in detections:
    boxes = detection.boxes
    print("hii")
    print(boxes)
    for box in boxes:
        print("hey")
        print(box.xyxy[0])
        x1, y1, x2, y2 = box.xyxy[0]
            # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Draw rectangle on the image array
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # If you have labels and confidence to display, add them here with cv2.putText()

# Display the image with bounding boxes using OpenCV
cv2.imshow("img", img)
cv2.waitKey(0)  # Wait for a key press to close the image window
cv2.destroyAllWindows()  # Ensure window is closed properly


