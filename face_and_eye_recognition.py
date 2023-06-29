import cv2
import sys
import time
import numpy as np


# VideoCaptureのインスタンスを作成する。
# 引数でカメラを選べれる。
cap = cv2.VideoCapture(0)

if cap.isOpened() is False:
    print("can not open camera")
    sys.exit()

# トラッカーとIDを初期化
tracker_dict = dict()
face_id_counter = 1
# Create an empty set for face IDs
IDs_set = set()

# 評価器を読み込み
# https://github.com/opencv/opencv/tree/master/data/haarcascades
cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
profile_face_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

def convert_x_to_angle(x, frame_width):
    return (x - frame_width / 2) / (frame_width / 2) * 180

while True:
    # VideoCaptureから1フレーム読み込む
    ret, frame = cap.read()

    # そのままの大きさだと処理速度がきついのでリサイズ
    frame = cv2.resize(frame, (int(frame.shape[1]), int(frame.shape[0])))

    # 処理速度を高めるために画像をグレースケールに変換したものを用意
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔検出
    frontal_facerect = cascade.detectMultiScale(
        gray,
        scaleFactor=1.11,
        minNeighbors=3,
        minSize=(100, 100)
    )

    # プロファイルフェイス検出
    profile_facerect = profile_face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.11,
        minNeighbors=3,
        minSize=(100, 100)
    )
    #... (前部のコード)

    # Check that both have detected faces, if not make them the correct shape
    if len(frontal_facerect) == 0:
        frontal_facerect = np.empty(shape=(0,4))
    if len(profile_facerect) == 0:
        profile_facerect = np.empty(shape=(0,4))

    # Combine the frontal and profile face rectangles into one list
    all_faces = np.concatenate((frontal_facerect, profile_facerect))

    for rect in all_faces:
        x, y, w, h = map(int, rect)
        # Checking if this face is already being tracked
        face_tracked = False
        for face_id, tracker in tracker_dict.items():
            success, bbox = tracker.update(frame)
            if success:
                (xt, yt, wt, ht) = [int(v) for v in bbox]
                # if the detected face and tracked face overlap significantly, they are the same face
                if (x <= xt + wt/2 <= x + w) and (y <= yt + ht/2 <= y + h):
                    face_tracked = True
                    IDs_set.discard(face_id)
                    break

        # If the face is not being tracked, create a new tracker
        if not face_tracked:
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x, y, w, h))
            tracker_dict[face_id_counter] = tracker
            face_id_counter += 1

    # ※HACK: 顔が消えたらIDを削除してしまっているが、一定の秒数残す処理を行うこともできる
    for face_id in IDs_set:
       del tracker_dict[face_id]
    
    IDs_set = set(tracker_dict.keys())  # 更新

    

    for face_id, tracker in tracker_dict.items():
        ok, bbox = tracker.update(frame)
        if ok:
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            angle = convert_x_to_angle(center_x, frame.shape[1])
            cv2.putText(frame, f'ID: {face_id}, Angle: {angle:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()