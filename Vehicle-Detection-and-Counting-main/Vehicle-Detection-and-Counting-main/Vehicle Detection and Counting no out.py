import cv2
import numpy as np

min_contour_width = 40  # Minimum contour width to consider for detection
min_contour_height = 40  # Minimum contour height to consider for detection
line_height = 550  # Detection line height to count vehicles
offset = 10
vehicles = 0  # Vehicle count

cap = cv2.VideoCapture('Video.mp4')  # Replace with your video path
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

previous_centroids = []
vehicle_types = {}

# Variables to store the list of vehicle dimensions
vehicle_sizes = []

def get_centroid(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

if cap.isOpened():
    ret, frame1 = cap.read()
else:
    ret = False
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while ret:
    d = cv2.absdiff(frame1, frame2)  # Frame differencing
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)  # Reduce noise and smooth the image
    ret2, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3)))  # Increase object size to improve contour detection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)  # Close small gaps in objects
    contours, h = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_centroids = []
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= min_contour_width and h >= min_contour_height:
            centroid = get_centroid(x, y, w, h)
            new_centroids.append({'centroid': centroid, 'counted': False, 'last_y': centroid[1], 'id': i})
            
            # Draw bounding box and centroid
            cv2.rectangle(frame1, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 0), 2)
            cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)

            # Save vehicle size for dynamic classification
            vehicle_sizes.append((w, h))

            # Classify vehicle based on size and aspect ratio
            aspect_ratio = w / h
            avg_size = np.mean([size[0] for size in vehicle_sizes]) if vehicle_sizes else 0
            if aspect_ratio > 2.5:  # Trucks typically have higher aspect ratio
                vehicle_types[i] = "Truck"
            elif w > avg_size:
                vehicle_types[i] = "Truck"
            else:
                vehicle_types[i] = "Car"

            # Display vehicle type
            vehicle_type = vehicle_types[i]
            cv2.putText(frame1, vehicle_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Match new centroids to previous centroids to track vehicles
    for new_obj in new_centroids:
        min_dist = 99999
        matched_obj = None
        for prev_obj in previous_centroids:
            dist = np.linalg.norm(np.array(new_obj['centroid']) - np.array(prev_obj['centroid']))
            if dist < 40:  # Adjust this distance threshold as needed
                if dist < min_dist:
                    min_dist = dist
                    matched_obj = prev_obj
        if matched_obj:
            prev_y = matched_obj['last_y']
            curr_y = new_obj['centroid'][1]
            # Count vehicle when crossing the line
            if (prev_y < line_height <= curr_y or prev_y > line_height >= curr_y) and not matched_obj['counted']:
                vehicles += 1
                matched_obj['counted'] = True
            new_obj['counted'] = matched_obj['counted']
            new_obj['last_y'] = curr_y
        else:
            new_obj['last_y'] = new_obj['centroid'][1]

    previous_centroids = new_centroids

    # Draw the counting line and display vehicle count
    cv2.line(frame1, (0, line_height), (frame_width, line_height), (0, 255, 0), 2)
    cv2.putText(frame1, f"Total Vehicles Detected: {vehicles}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2)
    cv2.imshow('Vehicle Detection (Live)', frame1)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break
    frame1 = frame2
    ret, frame2 = cap.read()

cap.release()
cv2.destroyAllWindows()
print("Processing finished!")
