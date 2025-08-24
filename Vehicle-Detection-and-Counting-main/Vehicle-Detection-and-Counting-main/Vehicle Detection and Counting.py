import cv2
import numpy as np

min_contour_width = 40  
min_contour_height = 40  
line_height = 550  
vehicles = 0

cap = cv2.VideoCapture('Video.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# Memory of previous centroids with their last y-coord and count status
previous_centroids = []

def get_centroid(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

if cap.isOpened():
    ret, frame1 = cap.read()
else:
    ret = False
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while ret:
    d = cv2.absdiff(frame1, frame2)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    ret2, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, h = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_centroids = []
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= min_contour_width and h >= min_contour_height:
            centroid = get_centroid(x, y, w, h)
            new_centroids.append({'centroid': centroid, 'counted': False, 'last_y': centroid[1]})
            cv2.rectangle(frame1, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 0), 2)
            cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)

            # 🚗 Vehicle classification (Size-based)
            if w > 120 or h > 120:  # You can adjust these thresholds
                vehicle_type = "Truck"
            else:
                vehicle_type = "Car"
            cv2.putText(frame1, vehicle_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Match new centroids to previous by minimum distance
    for new_obj in new_centroids:
        min_dist = 99999
        matched_obj = None
        for prev_obj in previous_centroids:
            dist = np.linalg.norm(np.array(new_obj['centroid']) - np.array(prev_obj['centroid']))
            if dist < 40:
                if dist < min_dist:
                    min_dist = dist
                    matched_obj = prev_obj
        if matched_obj:
            prev_y = matched_obj['last_y']
            curr_y = new_obj['centroid'][1]
            if (prev_y < line_height <= curr_y or prev_y > line_height >= curr_y) and not matched_obj['counted']:
                vehicles += 1
                matched_obj['counted'] = True
            new_obj['counted'] = matched_obj['counted']
            new_obj['last_y'] = curr_y
        else:
            new_obj['last_y'] = new_obj['centroid'][1]

    previous_centroids = new_centroids

    cv2.line(frame1, (0, line_height), (frame_width, line_height), (0, 255, 0), 2)
    cv2.putText(frame1, "Total Vehicle Detected: " + str(vehicles), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2)
    out.write(frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
print("Processing finished! Output video is 'output.mp4'")
