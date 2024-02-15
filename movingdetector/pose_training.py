import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ポーズ推定モデルの設定
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

prev_landmarks = None  # 前フレームのランドマーク

def detect_kick(landmarks, height):
    # 足が上がっているかを検出
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y

    # どちらかの足首が腰よりも高い位置にあれば、キックと判断
    if left_ankle < left_hip or right_ankle < right_hip:
        return "Kick"
    else:
        return None

def detect_dance(current_landmarks, prev_landmarks):
    # 動きが速いかを検出（Dance）
    if prev_landmarks is None:
        return None

    movement_threshold = 0.05  # 動きの閾値
    movement = 0
    for i, landmark in enumerate(current_landmarks):
        prev_landmark = prev_landmarks[i]
        # 全ランドマークについての動きの大きさを計算
        movement += abs(landmark.x - prev_landmark.x) + abs(landmark.y - prev_landmark.y) + abs(landmark.z - prev_landmark.z)

    if movement > movement_threshold:
        return "Dance"
    else:
        return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    action_texts = []

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        # ポーズの名前を特定
        kick_result = detect_kick(results.pose_landmarks.landmark, image.shape[0])
        if kick_result:
            action_texts.append(kick_result)
        
        dance_result = detect_dance(results.pose_landmarks.landmark, prev_landmarks)
        if dance_result:
            action_texts.append(dance_result)
        
        # 前フレームのランドマークを更新
        prev_landmarks = results.pose_landmarks.landmark

    # 複数のアクションを表示
    for i, action_text in enumerate(action_texts):
        cv2.putText(image, action_text, (50, 50 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('MediaPipe Pose', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

pose.close()
cap.release()
cv2.destroyAllWindows()
