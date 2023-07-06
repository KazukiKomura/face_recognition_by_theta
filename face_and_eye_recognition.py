import cv2
import numpy as np
import time


# Initialize cascade classifiers
cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
profile_face_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_profileface.xml')

# Create a dictionary to hold trackers and a counter for face IDs
trackers = dict()
last_update = dict()  # Holds the time of the last update for each tracker
face_id_counter = 1
TIMEOUT = 3  # Time in seconds after which a tracker is removed

def convert_x_to_angle(x, frame_width):
    return (x - frame_width / 2) / (frame_width / 2) * 180

def detect_faces(gray):
    frontal_faces = cascade.detectMultiScale(
        gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100)
    )

    profile_faces = profile_face_cascade.detectMultiScale(
        gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100)
    )

    if len(frontal_faces) == 0:
        frontal_faces = np.empty(shape=(0,4))
    if len(profile_faces) == 0:
        profile_faces = np.empty(shape=(0,4))

    return np.concatenate((frontal_faces, profile_faces))

def update_trackers(frame, gray):
    global face_id_counter
    faces = detect_faces(gray)
    for face in faces:
        x, y, w, h = map(int, face)

        face_tracked = False
        for face_id, tracker in trackers.items():
            success, bbox = tracker.update(frame)
            if success:
                (xt, yt, wt, ht) = [int(v) for v in bbox]
                if (x <= xt + wt/2 <= x + w) and (y <= yt + ht/2 <= y + h):
                    face_tracked = True
                    last_update[face_id] = time.time()  # Update the timestamp
                    break

        if not face_tracked:
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x, y, w, h))
            trackers[face_id_counter] = tracker
            last_update[face_id_counter] = time.time()  # Initialize the timestamp
            face_id_counter += 1

    # Remove trackers that haven't been updated in the last TIMEOUT seconds
    current_time = time.time()
    face_ids_to_remove = [face_id for face_id, update_time in last_update.items() if current_time - update_time > TIMEOUT]
    for face_id in face_ids_to_remove:
        del trackers[face_id]
        del last_update[face_id]

    return frame

def draw_frame(frame):
    for face_id, tracker in trackers.items():
        ok, bbox = tracker.update(frame)
        if ok:
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            center_x = x + w // 2
            cv2.circle(frame, (center_x, y), 5, (0, 0, 255), -1)
            angle = convert_x_to_angle(center_x, frame.shape[1])
            cv2.putText(frame, f'ID: {face_id}, Angle: {angle:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret,frame = cap.read()
        if not ret:
            print("Can't receive frame")
            break

        # Resize the frame for faster processing
        frame = cv2.resize(frame, (int(frame.shape[1]), int(frame.shape[0])))

        # Convert the frame to grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = update_trackers(frame, gray)
        frame = draw_frame(frame)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:  # Escape key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
