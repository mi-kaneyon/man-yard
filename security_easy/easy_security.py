import cv2
import torch
import mediapipe as mp
import numpy as np
import argparse
import warnings

# FutureWarning を抑制
warnings.filterwarnings("ignore", message=".*torch\.cuda\.amp\.autocast.*")

# --- コマンドライン引数でモデルを選択可能に ---
parser = argparse.ArgumentParser(description="Security Camera with Modular Detection Models")
parser.add_argument('--detector', choices=['yolov5s','yolov5l','yolov8n','yolov8s'], default='yolov5s',
                    help='Object detection model to use')
parser.add_argument('--device', choices=['cpu','cuda'], default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='Inference device')
parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
args = parser.parse_args()

device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'

class Detector:
    """
    Wrapper to load and run different detection models interchangeably.
    """
    def __init__(self, name, device, conf):
        self.name = name
        self.device = device
        self.conf = conf
        if name.startswith('yolov5'):
            self.model = torch.hub.load('ultralytics/yolov5', name, pretrained=True).to(device)
            self.model.conf = conf
            self.names = self.model.names
        elif name.startswith('yolov8'):
            from ultralytics import YOLO
            self.model = YOLO(f"{name}.pt")
            self.names = self.model.names
            try:
                self.model.model[-1].conf = conf
            except Exception:
                pass
        else:
            raise ValueError(f'Unknown model: {name}')

    def detect(self, frame):
        if self.name.startswith('yolov5'):
            results = self.model(frame)
            return results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
        else:
            results = self.model(frame)
            dets = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    dets.append([x1, y1, x2, y2, conf, cls])
            return np.array(dets)

# モジュール初期化
detector = Detector(args.detector, device, args.conf)
mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

# 動物クラス(COCOインデックス)
ANIMAL_CLASSES = {14,15,16,17,18,19,20,21,22,23}

# モザイク関数（座標チェック追加）

def mosaic_face(frame, x1, y1, x2, y2, scale=0.05):
    # 座標をフレーム内にクランプ
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    # サイズチェック
    if x2 <= x1 or y2 <= y1:
        return
    face = frame[y1:y2, x1:x2]
    # ROIが空の場合はスキップ
    if face.size == 0:
        return
    small = cv2.resize(face, (max(1, int((x2-x1)*scale)), max(1, int((y2-y1)*scale))), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)

# 不審動作判定

def is_suspicious(curr, prev, thresh=50):
    if not prev:
        return False
    dists = [np.linalg.norm(curr - np.array(p)) for p in prev]
    return min(dists) > thresh

# メインループ

def main():
    cap = cv2.VideoCapture(0)
    prev_centroids = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = detector.detect(frame)
        curr = []
        suspicious = []

        # 検出結果処理
        for x1,y1,x2,y2,conf,cls in dets:
            if conf < args.conf:
                continue
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            if cls == 0:  # 人
                cX, cY = (x1+x2)//2, (y1+y2)//2
                curr.append((cX,cY,x1,y1,x2,y2))
            elif cls in ANIMAL_CLASSES:
                label = detector.names[cls]
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # 不審者検出
        for c in curr:
            if is_suspicious(np.array([c[0],c[1]]), [(p[0],p[1]) for p in prev_centroids]):
                suspicious.append(c)

        # 人へのアノテーション
        for c in curr:
            x1,y1,x2,y2 = c[2:]
            if c in suspicious:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(frame,'WARNING',(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
            else:
                roi = frame[y1:y2, x1:x2]
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                face_res = mp_face.process(rgb)
                if face_res.detections:
                    for det in face_res.detections:
                        bb = det.location_data.relative_bounding_box
                        fx1 = int(bb.xmin*(x2-x1))+x1
                        fy1 = int(bb.ymin*(y2-y1))+y1
                        fw = int(bb.width*(x2-x1))
                        fh = int(bb.height*(y2-y1))
                        mosaic_face(frame, fx1, fy1, fx1+fw, fy1+fh)

        prev_centroids = [(c[0],c[1]) for c in curr]

        cv2.imshow('Security Camera', frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
