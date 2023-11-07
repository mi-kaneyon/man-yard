import cv2
import os
import time
import datetime

# 撮影間隔（秒）
capture_interval = 4  # 4秒ごとに設定

# 距離情報とサブディレクトリのリスト
distances = ['50cm', '1m', '1.5m', '2m']
base_dir = 'dataset'

# カメラデバイスを開く
cap = cv2.VideoCapture(0)

# トータルで撮影する画像の数
total_images_to_capture = 10  # 各距離ごとに撮影する画像の数

# 各距離ごとに撮影する画像の数をカウント
images_captured = {distance: 0 for distance in distances}

try:
    while True:  # 無限ループで継続的に撮影
        for distance in distances:
            # 各距離ごとのディレクトリを作成
            norm_dir = os.path.join(base_dir, distance, 'norm')
            os.makedirs(norm_dir, exist_ok=True)

            # 画像をキャプチャする
            ret, frame = cap.read()

            if ret and images_captured[distance] < total_images_to_capture:
                # 通常画像のファイル名（現在の日時を含む）
                norm_filename = f'norm_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.jpg'
                cv2.imwrite(os.path.join(norm_dir, norm_filename), frame)
                print(f'Normal image saved at {norm_dir}/{norm_filename}')
                images_captured[distance] += 1  # カウントを増やす
            else:
                print(f'Failed to capture image at {distance}')

            # 指定した秒数だけ待機
            time.sleep(capture_interval)

            # 全ての距離で十分な画像が撮影されたら終了
            if all(count >= total_images_to_capture for count in images_captured.values()):
                raise StopIteration

except StopIteration:
    # すべての画像が撮影されたらループを終了
    print("All images captured. Exiting...")

# カメラデバイスを解放
cap.release()

