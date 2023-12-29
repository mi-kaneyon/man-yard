# Import necessary libraries
import cv2
import numpy as np

# Dummy calibration data (replace with actual calibration results)
camera_matrix = np.array([[1000, 0, 320],
                          [0, 1000, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion


stereo = cv2.StereoSGBM_create(minDisparity=-1,
                               numDisparities=64,  # Must be divisible by 16
                               blockSize=5,
                               uniquenessRatio=10,
                               speckleWindowSize=100,
                               speckleRange=32,
                               disp12MaxDiff=1,
                               P1=8*3*5**2,
                               P2=32*3*5**2)


def resize_frame_fill(frame, target_size=(640, 480)):
    # Calculate the aspect ratio of the target size
    target_aspect = target_size[0] / target_size[1]
    # Calculate the aspect ratio of the original frame
    h, w = frame.shape[:2]
    original_aspect = w / h
    
    # Compare the aspect ratios
    if original_aspect > target_aspect:
        # Original is wider than target: scale by height
        scale_factor = target_size[1] / h
    else:
        # Original is taller than target: scale by width
        scale_factor = target_size[0] / w
    
    # Resize the frame with the scale factor
    resized_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

    # Create a new canvas and center the resized frame onto it
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    x_offset = (canvas.shape[1] - resized_frame.shape[1]) // 2
    y_offset = (canvas.shape[0] - resized_frame.shape[0]) // 2
    canvas[y_offset:y_offset+resized_frame.shape[0], x_offset:x_offset+resized_frame.shape[1]] = resized_frame
    
    return canvas

def adjust_colormap(disparity_map):
    # Assuming the disparity map values are normalized between 0 and 1
    # Placeholder logic: Darken the pixels as the disparity value decreases (which means they are further away)
    depth_colormap = 255 - (disparity_map / disparity_map.max() * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_RAINBOW)
    return depth_colormap


# Compute disparity map from preprocessed left and right images
def compute_disparity_map(left_image, right_image):
    left_processed = preprocess_image(left_image)
    right_processed = preprocess_image(right_image)
    disparity_map = stereo.compute(left_processed, right_processed).astype(np.float32) / 16.0
    
# Initialize ORB detector
orb = cv2.ORB_create()

# Detect and compute keypoints and descriptors
def detect_and_describe(image):
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

# Match descriptors between two images
def match_keypoints(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def process_video_stream():
    # Initialization of previous frame variables for optical flow and temporal filtering
    prev_gray = None  # Previous grayscale frame
    prev_disparity_map = None  # Previous disparity map


def temporal_filtering(current_map, previous_map, alpha=0.9):
    """
    Apply temporal filtering to smooth the disparity map over time.
    :param current_map: The current disparity map.
    :param previous_map: The previous disparity map.
    :param alpha: The weight for the current frame. (0 < alpha < 1)
    :return: Temporally filtered disparity map.
    """
    if previous_map is None:
        # If there is no previous map, return the current map
        return current_map
    else:
        # Calculate the weighted sum of the current and previous maps
        return cv2.addWeighted(current_map, alpha, previous_map, 1 - alpha, 0)

def calculate_optical_flow(prev_gray, gray):
    """
    Calculate the optical flow between two consecutive grayscale images.
    :param prev_gray: The previous frame in grayscale.
    :param gray: The current frame in grayscale.
    :return: The points of the detected flow.
    """
    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Find initial points for tracking (ShiTomasi)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    if p0 is not None:
        # Calculate optical flow (Lucas-Kanade)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        return good_new, good_old
    else:
        # If no points found, return None
        return None, None




def process_video_stream():
    # Initialization of previous frame variables for optical flow and temporal filtering
    global prev_gray, prev_disparity_map
    prev_gray = None  # Previous grayscale frame
    prev_disparity_map = None  # Previous disparity map
    
    # The rest of the function's logic should go here
    # This is where you would implement the processing of the video stream, including optical flow
    # For now, it will be a placeholder since the specific implementation details are not provided
    pass
# カメラデバイスの初期化
cap1 = cv2.VideoCapture(0)  # カメラ1
cap2 = cv2.VideoCapture(2)  # カメラ2



# 前のフレームの保存用変数
prev_gray = None
prev_disparity_map = None


while True:
    # 両方のカメラからフレームを取得
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # フレームが正しく取得できたか確認
    if not ret1 or not ret2:
        print("Failed to grab frames")
        break

    
    # Resize frame to match the desired size of 640x480
    frame1 = resize_frame_fill(frame1)
    frame2 = resize_frame_fill(frame2)
    # フレームの前処理
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # ステレオマッチングでディスパリティマップを計算
    disparity_map_left = stereo.compute(gray1, gray2).astype(np.float32) / 16.0
    disparity_map_right = stereo.compute(gray2, gray1).astype(np.float32) / 16.0

    # WLSフィルタを作成
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.5)

    # WLSフィルタを適用
    filtered_disparity = wls_filter.filter(disparity_map_left, gray1, disparity_map_right=disparity_map_right)

    # バイラテラルフィルタを適用
    filtered_disparity = cv2.bilateralFilter(filtered_disparity, d=5, sigmaColor=75, sigmaSpace=75)

    # ディスパリティマップをヒートマップに変換して表示
    heatmap = cv2.applyColorMap(cv2.convertScaleAbs(filtered_disparity, alpha=255.0/64), cv2.COLORMAP_JET)
    heatmap = adjust_colormap(heatmap)

    heatmap_resized = cv2.resize(heatmap, (640, 480))
    cv2.imshow('Disparity Map', cv2.resize(heatmap, (640, 480)))

    # 'q'キーが押されたらループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラデバイスのリリースとウィンドウの破棄
cap1.release()
cap2.release()
cv2.destroyAllWindows()
