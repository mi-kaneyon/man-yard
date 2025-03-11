import open3d as o3d
import numpy as np
import cv2
import glob
import os

# フォルダ設定
depth_maps_dir = "stereo/depth_maps"
color_images_dir = "stereo/front"
pointcloud_output_dir = "stereo/pointclouds"

os.makedirs(pointcloud_output_dir, exist_ok=True)

# カメラ内部パラメータ（calibrate.pyの結果を使用）
fx, fy = 456.393, 461.379
cx, cy = 347.329, 151.266

# depth map画像と元のカラー画像を取得
depth_files = sorted(glob.glob(os.path.join(depth_maps_dir, "depth_*.png")))
color_files = sorted(glob.glob(os.path.join(color_images_dir, "*.jpg")))

for idx, (depth_file, color_file) in enumerate(zip(depth_files, color_files)):
    # 深度マップを読み込み (グレースケール)
    depth = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    color = cv2.cvtColor(cv2.imread(color_file), cv2.COLOR_BGR2RGB)

    # Open3D用にデータを整形
    depth_scaled = (depth * 1000).astype(np.uint16)  # 深度スケール(mm単位にするため)
    depth_o3d = o3d.geometry.Image(depth_scaled)
    color_o3d = o3d.geometry.Image(color)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1000.0,  # mm単位
        depth_trunc=3000.0,  # 3メートル以上は無視する場合
        convert_rgb_to_intensity=False
    )

    # カメラキャリブレーションで取得した内部パラメータ（再確認して入力）
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=color.shape[1],
        height=color.shape[0],
        fx=456.393, fy=461.379,
        cx=347.329, cy=151.266
    )

    # 点群に変換
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # 点群データを保存
    pointcloud_file = os.path.join(pointcloud_output_dir, f"pointcloud_{idx}.ply")
    o3d.io.write_point_cloud(pointcloud_file, pcd)

    print(f"3D点群を保存しました: {pointcloud_file}")

    # 点群を表示（任意）
    o3d.visualization.draw_geometries([pcd])
