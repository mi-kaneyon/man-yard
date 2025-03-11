import cv2
import numpy as np
import glob
import sys

CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)

objpoints = []
imgpoints = []

images = glob.glob('camera1/*.jpg')

# 読み込んだ画像パスの確認
if len(images) == 0:
    print("画像が見つかりません。画像パスを再確認してください。")
    sys.exit(1)

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"{fname} は読み込みに失敗しました。")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
    else:
        print(f"{fname} ではチェスボードを検出できませんでした。")

if len(objpoints) == 0:
    print("有効な画像がありません。再撮影してください。")
    sys.exit(1)

ret, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("カメラ行列:\n", cameraMatrix)
print("歪み係数:\n", distCoeffs)
