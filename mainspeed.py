import cv2
import numpy as np
import time
import math

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Vehicle classes
VEHICLE_CLASSES = ["car", "motorbike", "bus", "truck","traffic light", "person"]

# Capture video
# Load video file
content=""
with open('demofile2.txt', 'r') as file:
    # Read the content of the file
    content = file.read()
    
# Display the content
print(content)
video_path = content
cap = cv2.VideoCapture(video_path)  # Change to 0 for real-time webcam
fps = cap.get(cv2.CAP_PROP_FPS)  # Frame rate
scale_factor = 0.05  # Approximate scale factor (adjust based on camera setup)

prev_vehicle_positions = {}  # Store previous vehicle positions
frame_id = 0  # Frame counter

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    height, width, _ = frame.shape
    start_time = time.time()

    # Prepare YOLO input
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    # Store detected vehicle positions
    vehicle_positions = {}

    # Process detections
    boxes, confidences, class_ids = [], [], []
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] in VEHICLE_CLASSES:
                # Object detected
                center_x, center_y, w, h = map(int, detection[:4] * [width, height, width, height])
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # Store vehicle position
                vehicle_id = len(vehicle_positions)  # Assign an ID
                vehicle_positions[vehicle_id] = (center_x, center_y)

    # Apply Non-Maximum Suppression (NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and compute speed
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        center_x, center_y = x + w // 2, y + h // 2
        label = f"{classes[class_ids[i]]}"
        color = (0, 0, 255)  # Red for vehicles

        # Speed estimation
        speed = 0
        if i in prev_vehicle_positions:
            prev_x, prev_y = prev_vehicle_positions[i]
            pixel_distance = math.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
            speed = (pixel_distance * scale_factor * fps) * 3.6  # Convert to km/h

        # Distance estimation between vehicles
        min_distance = float("inf")
        disalter=""
        for j, (other_x, other_y) in vehicle_positions.items():
            if i != j:
                distance = math.sqrt((center_x - other_x) ** 2 + (center_y - other_y) ** 2) * scale_factor
                min_distance = min(min_distance, distance) *100
                if min_distance>10:
                    disalter="Normal"
                if min_distance<10 and min_distance>5:
                    disalter="Careful Driving"
                if min_distance<5:
                    disalter="Alert Alert"
                
                    

        # Draw bounding box and info
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {speed:.1f} km/h", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"Dist: {min_distance:.1f}m "+disalter, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update previous positions
    prev_vehicle_positions = vehicle_positions.copy()

    # Show result
    cv2.imshow("YOLOv3 Vehicle Detection with Speed & Distance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
