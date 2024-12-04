# BitPyre Project (prototype)

import cv2
import numpy as np

def load_yolo_model(cfg_path, weights_path, names_path):
    # Load YOLO model
    net = cv2.dnn.readNet(weights_path, cfg_path)
    with open(names_path, 'r') as f:
        classes = f.read().strip().split('\n')
    return net, classes

def detect_objects(frame, net, classes, conf_threshold=0.5, nms_threshold=0.4):
    # Convert the frame to a blob for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    h, w = frame.shape[:2]
    boxes, confidences, class_ids = [], [], []

    # Extract detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                box = detection[0:4] * np.array([w, h, w, h])
                center_x, center_y, width, height = box.astype('int')
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression to reduce overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    results = [(boxes[i], confidences[i], class_ids[i]) for i in indices.flatten()]
    return results

def detect_fire(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color range for fire (adjust as needed)
    lower_fire = np.array([10, 100, 100])  # Lower bound of fire color
    upper_fire = np.array([35, 255, 255])  # Upper bound of fire color

    # Create a mask for the fire color
    mask = cv2.inRange(hsv, lower_fire, upper_fire)

    # Perform morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fire_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Ignore small detections
            x, y, w, h = cv2.boundingRect(contour)
            fire_boxes.append((x, y, w, h))
    return fire_boxes

def main():
    # Paths to YOLO files (Change this by your filename path)
    cfg_path = "C:/Users/Arron/Documents/Portfolio Projecr/Python Facial Recognition camera/yolov3.cfg"
    weights_path = "C:/Users/Arron/Documents/Portfolio Projecr/Python Facial Recognition camera/yolov3.weights"
    names_path = "C:/Users/Arron/Documents/Portfolio Projecr/Python Facial Recognition camera/coco.names"

    # Load YOLO model
    net, classes = load_yolo_model(cfg_path, weights_path, names_path)

    # Start video capture (camera or video file)
    cap = cv2.VideoCapture(0)  # Replace 0 with a video file path for a file or pede niyo rin lagyan ng video file to test it out hehe

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame captured. Exiting...")
            break

        # Detect fire in the frame
        fire_boxes = detect_fire(frame)

        # Detect objects in the frame
        results = detect_objects(frame, net, classes)

        # Draw fire detections
        for (x, y, w, h) in fire_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Fire", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Border box ni PAPA YOLO HAHAHA
        for (box, confidence, class_id) in results:
            x, y, w, h = box
            label = f"{classes[class_id]}: {confidence:.2f}"
            color = (0, 255, 0) if classes[class_id] == "person" else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the frame
        cv2.imshow("Fire and Human Detection", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
    
# Line 72 change filename path depende ng sanyo
# install this packages
# pip install opencv-python <<<< for camera
# pip install numpy <<<< numerical computing
# pip install matplotlib <<<<< for visualization ng format
# pip install opencv-python numpy matplotlib <<<<< you can do this for shortcut alsooo




