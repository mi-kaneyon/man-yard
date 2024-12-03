import cv2
import mediapipe as mp
from PIL import Image, ImageEnhance
import numpy as np

def enhance_resize_and_white_background(input_path, output_path, scale_factor=2.0, sharpness=2.0):
    # MediaPipeのセットアップ
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        
        # 画像を読み込み、RGBに変換
        img = cv2.imread(input_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # MediaPipeで人物領域を検出
        results = selfie_segmentation.process(img_rgb)
        mask = results.segmentation_mask

        # マスクを使って背景を白に設定
        condition = np.stack((mask,) * 3, axis=-1) > 0.5
        white_background = np.ones(img_rgb.shape, dtype=np.uint8) * 255
        img_with_white_bg = np.where(condition, img_rgb, white_background)

        # Pillowに変換してシャープネス調整
        pil_img = Image.fromarray(img_with_white_bg)
        enhancer = ImageEnhance.Sharpness(pil_img)
        img_sharp = enhancer.enhance(sharpness)  # シャープネスの強さをパラメータ化
        
        # サイズを指定倍率で拡大（少数対応）
        new_size = (int(img_sharp.width * scale_factor), int(img_sharp.height * scale_factor))
        img_resized = img_sharp.resize(new_size, Image.LANCZOS)
        
        # 保存
        img_resized.save(output_path)
        print(f"変換が完了しました。保存先: {output_path}")

# 使用例
input_path = "blueprevious.png"  # 元の画像ファイルパス
output_path = "whiteback.png"  # 保存先のファイルパス
enhance_resize_and_white_background(input_path, output_path, scale_factor=2.5, sharpness=2.0)  # シャープネスと拡大率を調整
