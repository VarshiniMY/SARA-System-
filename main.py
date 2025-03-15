import cv2
import numpy as np

# Load YOLO model
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Object detection function
def detect_objects(frame, net, output_layers):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

# Estimate distance (simple example based on object height/width)
def estimate_distance(box_width, known_width, focal_length):
    # Distance estimation formula: (Real object width * Focal Length) / Object width in image
    if box_width == 0:
        return 0
    distance = (known_width * focal_length) / box_width
    return distance

# Draw bounding boxes and display distance estimation
def draw_boxes(frame, boxes, confidences, class_ids, classes, focal_length, known_width):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            distance = estimate_distance(w, known_width, focal_length)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {distance:.2f}m', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

# Load class labels
def load_classes():
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# Main function to run object detection with distance estimation
def main():
    net, output_layers = load_yolo()
    classes = load_classes()
    
    # Set the known width of the object and the focal length
    known_width = 0.5  # Example: width of a car in meters
    focal_length = 600  # Example: focal length in pixels, depends on the camera

    cap = cv2.VideoCapture(0)  # Capture from webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, confidences, class_ids = detect_objects(frame, net, output_layers)
        frame = draw_boxes(frame, boxes, confidences, class_ids, classes, focal_length, known_width)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
