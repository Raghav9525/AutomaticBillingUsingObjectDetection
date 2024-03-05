import threading
import cv2
from ultralytics import YOLO  # Assuming YOLO is correctly imported
import cvzone
import constant  # This module needs to contain 'classNames'
import torch
import math

def detections(stop_detection_flag):
    # Assuming classNames is defined in 'constant'
    classNames = constant.classNames

    # Path to your trained YOLOv8 model weights
    model_path = "./models/best7.pt"  # Update with your model's path
    model = YOLO(model_path)

    # Initialize video capture from the default camera
    cap = cv2.VideoCapture(0)

    frame_ids = {
        "samosaCount": [],
    }
    response_dict = {}

    try:
        while True:
            # Check if the stop signal is set
            if stop_detection_flag.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            samosaCount = 0

            with torch.no_grad():
                # Assuming the model's prediction method returns detections directly
                detections = model(frame)

            for detection in detections:
                boxes = detection.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]

                    # Convert coordinates to integers
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=9, rt=2, colorR=(255, 0, 0))
                    cls = int(box.cls[0])
                    CurrentClass = classNames[cls]

                    if CurrentClass == "samosa" and conf > 0.5:
                        samosaCount += 1
                        cvzone.putTextRect(frame, f"Samosa {conf:.2f}", (x1, y1 - 10), scale=1, thickness=2, offset=0)

                print(f"samosa: {samosaCount}")
                frame_ids["samosaCount"].append(samosaCount)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error in detection thread: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for item, counts in frame_ids.items():
            response_dict[item] = max(counts) if counts else 0

        print(response_dict)
        # Optionally, signal that processing is complete
        return response_dict

# Example usage:
# Create a threading.Event() for stopping the detection loop
stop_detection_flag = threading.Event()
# Run your detection function (potentially in a separate thread)
detections(stop_detection_flag)
