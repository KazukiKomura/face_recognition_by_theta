import cv2
import sys


# VideoCaptureのインスタンスを作成する。
# 引数でカメラを選べれる。
cap = cv2.VideoCapture(1)

if cap.isOpened() is False:
    print("can not open camera")
    sys.exit()

# 評価器を読み込み
# https://github.com/opencv/opencv/tree/master/data/haarcascades
cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def mosaic_area(src, x, y, width, height, ratio=0.1):
    dst = src.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst

while True:
    # VideoCaptureから1フレーム読み込む
    ret, frame = cap.read()

    # そのままの大きさだと処理速度がきついのでリサイズ
    frame = cv2.resize(frame, (int(frame.shape[1]), int(frame.shape[0])))

    # 処理速度を高めるために画像をグレースケールに変換したものを用意
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔検出
    facerect = cascade.detectMultiScale(
        gray,
        scaleFactor=1.11,
        minNeighbors=3,
        minSize=(100, 100)
    )

    if len(facerect) != 0:
        for x, y, w, h in facerect:
            # 顔の部分(この顔の部分に対して目の検出をかける)
            face_gray = gray[y: y + h, x: x + w]

            # くり抜いた顔の部分を表示(処理には必要ない。ただ見たいだけ。)
            #show_face_gray = cv2.resize(face_gray, (int(gray.shape[1]), int(gray.shape[0])))
            #cv2.imshow('face', show_face_gray)

            # 顔の部分から目の検出
            eyes = eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.11, # ここの値はPCのスペックに依存するので適宜修正してください
                minNeighbors=3,
                minSize=(15, 15)
            )

            center_x = x + w // 2
            center_y = y + h // 2

            # 顔の中心点を画面に表示
            cv2.circle(frame, (center_x, center_y), radius=5, color=(255, 0, 0), thickness=-1)
            cv2.putText(frame, f'Center: ({center_x}, {center_y})', (center_x + 10, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            '''
            if len(eyes) == 0:
                # 目が閉じられたとみなす
                cv2.putText(
                    frame,
                    'close your eyes',
                    (x, y - 10), # 位置を少し調整
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255,0),
                    2,
                    cv2.LINE_AA
                )
            else:
                for (ex, ey, ew, eh) in eyes:
                    # 目の部分にモザイク処理
                    frame = mosaic_area(
                        frame,
                        int((x + ex) - ew / 2),
                        int(y + ey),
                        int(ew * 2.5),
                        eh
                    )
            '''
            # 顔検出した部分に枠を描画
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                thickness=2
            )

    cv2.imshow('frame', frame)

    # キー入力を1ms待って、k が27（ESC）だったらBreakする
    k = cv2.waitKey(1)
    if k == 27:
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()