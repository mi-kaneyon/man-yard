import cv2
import os
import time

output_dir = 'camera1'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("カメラが開けません。")
    exit()

total_images = 30  # 撮影枚数
countdown_seconds = 3  # 撮影間隔（秒）

print("カメラ準備完了。撮影を開始するにはスペースキーを押してください。")
while True:
    ret, frame = cap.read()
    if not ret:
        print("フレーム取得失敗")
        break

    cv2.putText(frame, "Press SPACE to start capturing", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

for i in range(total_images):
    for countdown in range(countdown_seconds, 0, -1):
        ret, frame = cap.read()
        cv2.putText(frame, f'Capturing in {countdown} sec...', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Camera', frame)
        cv2.waitKey(1000)

    ret, frame = cap.read()
    if ret:
        filename = f"{output_dir}/chessboard_{i+1}.jpg"
        cv2.imwrite(filename, frame)
        print(f"{filename} を保存しました。")

cap.release()
cv2.destroyAllWindows()
print("撮影が完了しました。")
