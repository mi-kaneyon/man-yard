import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F

# CUDAが使用可能かどうかを確認
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# セグメンテーションモデルのロード（Mask R-CNN）
model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
model.eval()

def get_segmentation_masks(frame_tensor):
    with torch.no_grad():
        output = model(frame_tensor)
    
    masks = output[0]['masks']
    thresholded_masks = masks > 0.5
    
    return thresholded_masks.squeeze().detach().cpu().numpy()

def apply_rainbow_effect_segmentation(frame, masks):
    frame_with_rainbow = frame.copy()

    if masks.size > 0:  # マスクが存在し、空でないことを確認
        # マスクを適切なデータタイプに変換（例: boolからuint8へ）
        mask = masks[0].astype(np.uint8)  # マスクがbool型の場合、uint8に変換

        # マスクのリサイズ
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        # HSV色空間に変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # マスクが1のピクセルに対して虹色を適用
        h[mask_resized == 1] = np.mod(180 * np.arange(np.sum(mask_resized == 1)) / np.sum(mask_resized == 1), 180).astype(np.uint8)
        s[mask_resized == 1] = 255
        v[mask_resized == 1] = 255

        hsv = cv2.merge([h, s, v])
        frame_with_rainbow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return frame_with_rainbow



print("Press 'q' to quit.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # フレームの前処理
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = F.to_tensor(frame_rgb).unsqueeze(0).to(device)

    # セグメンテーションマスクを取得
    masks = get_segmentation_masks(frame_tensor)

    # マスクが存在する場合のみ虹色の残像を適用
    if masks.any():
        frame_with_rainbow = apply_rainbow_effect_segmentation(frame, masks)
    else:
        frame_with_rainbow = frame  # マスクがなければ元のフレームをそのまま使用

    # 結果を表示
    cv2.imshow('Rainbow Trail', frame_with_rainbow)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
