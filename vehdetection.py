import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open video capture (replace with your video file or 0 for webcam)
cap = cv2.VideoCapture("Manhattan_detection.avi")  # Replace with 0 to use the webcam

# Loop through the video frames
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if no frames are left
    
    height, width, channels = frame.shape
    
    # Convert the frame to a blob
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
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the frame with bounding boxes
    cv2.imshow("Vehicle Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close the windows
cap.release()
cv2.destroyAllWindows()
