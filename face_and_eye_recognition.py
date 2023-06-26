import cv2
import sys
import numpy as np


# VideoCaptureのインスタンスを作成する。
# 引数でカメラを選べれる。
cap = cv2.VideoCapture(1)

if cap.isOpened() is False:
    print("can not open camera")
    sys.exit()

# 評価器を読み込み
# https://github.com/opencv/opencv/tree/master/data/haarcascades
cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
profile_face_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def mosaic_area(src, x, y, width, height, ratio=0.1):
    dst = src.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst
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
    # Choose the face with the larger area if both are detected
    if len(frontal_facerect) != 0 and len(profile_facerect) != 0:
        if frontal_facerect[0][2] * frontal_facerect[0][3] > profile_facerect[0][2] * profile_facerect[0][3]:
            facerect = frontal_facerect
        else:
            facerect = profile_facerect
    elif len(frontal_facerect) != 0:
        facerect = frontal_facerect
    elif len(profile_facerect) != 0:
        facerect = profile_facerect
    else:
        facerect = []
    # handle both frontal and profile faces separately
    #for facerect in [facerect, profile_facerect]:
    if len(facerect) != 0:
        for x, y, w, h in facerect:

            center_x = x + w // 2
            center_y = y + h // 2

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            angle = convert_x_to_angle(center_x, frame.shape[1])
            cv2.putText(frame, f'Angle: {angle:.2f}', (center_x + 10, center_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
