import cv2

# カメラインデックスを指定。通常、0は内蔵カメラ、1以降が他のカメラ
# OBSの仮想カメラなら1や2等を試す
cap = cv2.VideoCapture(1)

while(True):
    # フレームを取得
    ret, frame = cap.read()

    # フレームを表示
    cv2.imshow('frame', frame)

    # 'q'キーが押されたらループから抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャを終了
cap.release()
cv2.destroyAllWindows()
