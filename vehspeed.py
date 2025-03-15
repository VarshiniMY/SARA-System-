import cv2
import numpy as np
import time

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open video capture (replace with your video file or 0 for webcam)
cap = cv2.VideoCapture("Manhattan_detection.avi")  # Replace with 0 for webcam

# Set up the video writer to save the processed video
frame_width = int(cap.get(3))  # Width of the frame
frame_height = int(cap.get(4))  # Height of the frame
out = cv2.VideoWriter('output_video_with_speed.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to calculate speed (in km/h for example, assuming frame rate of 30 FPS)
def calculate_speed(distance, fps):
    # Assuming each pixel is 0.1 meter (adjust based on your scene setup)
    pixel_to_meter = 0.1
    # Time between frames in seconds
    time_per_frame = 1 / fps
    # Speed in m/s
    speed = distance * pixel_to_meter / time_per_frame
    # Convert to km/h
    speed_kmh = speed * 3.6
    return speed_kmh

# Start processing the video
fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
prev_positions = {}  # Dictionary to store the previous positions of vehicles

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if no frames are left
    
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    # Feed the blob to the network
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists to hold detection data
    class_ids = []
    confidences = []
    boxes = []

    # Loop over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5 and classes[class_id] in ['car', 'bus', 'truck']:  # Filter for vehicles
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to remove redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes and labels on the frame
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green box for vehicle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Vehicle position for speed calculation
            vehicle_id = f"{label}_{i}"  # Unique ID for each vehicle based on its index and label
            if vehicle_id in prev_positions:
                prev_x, prev_y = prev_positions[vehicle_id]
                # Calculate the distance the vehicle has moved
                distance = calculate_distance(prev_x, prev_y, x + w / 2, y + h / 2)
                # Calculate the speed
                speed = calculate_speed(distance, fps)
                # Show speed on the frame
                cv2.putText(frame, f"Speed: {speed:.2f} km/h", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Update the previous position of the vehicle
            prev_positions[vehicle_id] = (x + w / 2, y + h / 2)

    # Show the frame with bounding boxes and speeds
    cv2.imshow("Vehicle Detection with Speed", frame)
    
    # Write the frame to the output video
    out.write(frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer, then close windows
cap.release()
out.release()
cv2.destroyAllWindows()
