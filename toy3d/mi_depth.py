import cv2
import torch
import numpy as np
import glob
import os

# GPUが使える場合はGPUを利用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MiDaSモデルをロード
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# 正面カメラの画像を取得（front/*.jpg）
images = sorted(glob.glob("stereo/front/*.jpg"))
output_dir = "stereo/depth_maps"
os.makedirs(output_dir, exist_ok=True)

for i, image_path in enumerate(images):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # MiDaS用に変換
    input_batch = transform(img).to(device)

    # 深度推定
    with torch.no_grad():
        prediction = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bilinear",
        align_corners=False
    ).squeeze().cpu().numpy()

    # 深度マップを0～255に正規化
    depth_map = cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    depth_filename = os.path.join(output_dir, f"depth_{i}.png")
    cv2.imwrite(depth_filename, depth_map)
    print(f"深度マップを保存しました: {depth_filename}")
