# faceplay.py
# Always-on GUI, human region matting, and Good/Bad classification using MediaPipe Tasks (blendshapes).
# - Matting: Robust Video Matting (TorchScript) -> fallback to convex hull from MediaPipe Face Landmarker.
# - Expressions: MediaPipe Face Landmarker (Tasks API) with blendshapes.
# - Scoring: Smile + Eye openness from blendshapes -> EMA smoothing -> hysteresis.
# - Visualization: Bright gradient background; person color brightens with score.
#   Labels: ["Happiness","Justice"] or ["Evil"]

import os
import sys
import time
import importlib
import subprocess
from pathlib import Path

import numpy as np
import cv2
import torch  # needed for TorchScript and @torch.no_grad

# ---------- Paths & constants ----------
HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# RVM TorchScript (official)
RVM_CKPT = CKPT_DIR / "rvm_mobilenetv3_fp32.torchscript"
RVM_URL  = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.torchscript"

# MediaPipe Face Landmarker .task (blendshapes-capable)
MP_TASK = CKPT_DIR / "face_landmarker_v2_with_blendshapes.task"
MP_TASK_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Good/Bad decision parameters
GOOD_ON   = 55.0    # enter Good above this (smoothed score)
GOOD_OFF  = 45.0    # exit Good below this
EMA_ALPHA = 0.20    # smoothing factor (0..1), higher = more reactive

# ---------- Small helpers ----------
def ensure_package(mod_name: str, pip_name: str = None, extras: str = ""):
    """Ensure a Python module is importable; if not, install via pip."""
    pip_name = pip_name or mod_name
    try:
        return importlib.import_module(mod_name)
    except ImportError:
        print(f"[INFO] Installing {pip_name}{extras} ...")
        cmd = [sys.executable, "-m", "pip", "install", f"{pip_name}{extras}"]
        subprocess.run(cmd, check=True)
        return importlib.import_module(mod_name)

def safe_download(url: str, dst: Path, chunk: int = 1 << 20):
    """Download a file if missing."""
    if dst.exists() and dst.stat().st_size > 0:
        return
    import urllib.request
    print(f"[INFO] Downloading: {url}")
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
    print(f"[INFO] Saved: {dst}")

def pick_device() -> str:
    """Prefer CUDA, then Apple MPS, else CPU."""
    try:
        if torch.cuda.is_available():
            print("[INFO] Selected device: cuda")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[INFO] Selected device: mps")
            return "mps"
    except Exception:
        pass
    print("[INFO] Selected device: cpu")
    return "cpu"

# ---------- Matting backends ----------
class MattingBackend:
    def infer_alpha(self, frame_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class RVMBackend(MattingBackend):
    """Robust Video Matting (TorchScript, MobileNetV3)."""
    def __init__(self, device: str):
        safe_download(RVM_URL, RVM_CKPT)
        print("[INFO] Loading RVM TorchScript checkpoint ...")
        self.device = device
        self.model = torch.jit.load(str(RVM_CKPT), map_location=device)
        self.model.eval()
        self.rec = [None] * 4  # recurrent states

    @torch.no_grad()
    def infer_alpha(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ten = torch.from_numpy(frame_rgb).float().to(self.device) / 255.0
        ten = ten.permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
        fgr, pha, *self.rec = self.model(ten, *self.rec, downsample_ratio=1.0)
        return pha[0, 0].clamp(0, 1).detach().cpu().numpy()

# ---------- MediaPipe Tasks: Face Landmarker ----------
mp = ensure_package("mediapipe")
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

def init_face_landmarker(num_faces: int = 1):
    """Initialize Face Landmarker in VIDEO mode with blendshapes."""
    safe_download(MP_TASK_URL, MP_TASK)
    base_opts = mp_python.BaseOptions(model_asset_path=str(MP_TASK))
    landmarker_opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        num_faces=num_faces,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        running_mode=mp_vision.RunningMode.VIDEO,  # VIDEO mode for detect_for_video
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(landmarker_opts)
    return landmarker

def compute_goodness_from_blendshapes(blendshapes) -> float:
    """
    0..100 'goodness' score from blendshapes:
    - Smile: avg(mouthSmileLeft/right)
    - Eye openness: 1 - avg(eyeBlinkLeft/right)
    """
    if not blendshapes:
        return 0.0
    d = {b.category_name: b.score for b in blendshapes}
    smile = (d.get("mouthSmileLeft", 0.0) + d.get("mouthSmileRight", 0.0)) / 2.0
    eye_open = 1.0 - (d.get("eyeBlinkLeft", 0.0) + d.get("eyeBlinkRight", 0.0)) / 2.0
    score = (0.7 * smile + 0.3 * eye_open) * 100.0
    return float(max(0.0, min(100.0, score)))

class GoodBadState:
    """EMA smoothing + hysteresis for Good/Bad."""
    def __init__(self, on=GOOD_ON, off=GOOD_OFF, alpha=EMA_ALPHA):
        self.on = on
        self.off = off
        self.alpha = alpha
        self.ema = 0.0
        self.is_good = False

    def update(self, score: float):
        self.ema = (1 - self.alpha) * self.ema + self.alpha * score
        if not self.is_good and self.ema >= self.on:
            self.is_good = True
        elif self.is_good and self.ema <= self.off:
            self.is_good = False
        return self.ema, self.is_good

def labels_from_state(is_good: bool):
    return (["Happiness", "Justice"] if is_good else ["Evil"])

# ---------- Fallback matte from Landmarker ----------
def alpha_from_landmarks(result, h, w):
    """Approximate face matte as convex hull over all face landmarks."""
    alpha = np.zeros((h, w), dtype=np.float32)
    if not result.face_landmarks:
        return alpha
    for lm_list in result.face_landmarks:
        pts = np.array([[int(p.x * w), int(p.y * h)] for p in lm_list], dtype=np.int32)
        if len(pts) >= 3:
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(alpha, hull, 1.0)
    alpha = cv2.GaussianBlur(alpha, (21, 21), 0)
    if alpha.max() > alpha.min():
        alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min())
    return alpha

# ---------- Visualization helpers ----------
def draw_keypoints_from_landmarker(comp, result, color):
    """Draw simple facial circles (eyes center, nose tip, mouth center)."""
    if not result.face_landmarks:
        return comp
    h, w = comp.shape[:2]
    lm = result.face_landmarks[0]

    # Indices compatible with FaceMesh/Face Landmarker topology
    nose_tip_idx = 1
    mouth_l_idx, mouth_r_idx = 61, 291
    mouth_top_idx, mouth_bot_idx = 13, 14
    left_eye_center_idx, right_eye_center_idx = 159, 386

    def P(i):
        return (int(lm[i].x * w), int(lm[i].y * h))

    nose = P(nose_tip_idx)
    mouth_c = (int((lm[mouth_l_idx].x + lm[mouth_r_idx].x) * 0.5 * w),
               int((lm[mouth_top_idx].y + lm[mouth_bot_idx].y) * 0.5 * h))
    left_eye = P(left_eye_center_idx)
    right_eye = P(right_eye_center_idx)

    for p in (left_eye, right_eye, nose, mouth_c):
        cv2.circle(comp, p, 6, color, 2, cv2.LINE_AA)
    return comp

def make_linear_gradient(h, w, c1=(255, 220, 180), c2=(180, 220, 255)):
    """Bright diagonal linear gradient from c1 (TL) to c2 (BR). BGR colors."""
    yy, xx = np.mgrid[0:h, 0:w]
    t = ((xx + yy) / float(w + h)).astype(np.float32)  # 0..1 along diagonal
    c1 = np.array(c1, dtype=np.float32)[None, None, :]
    c2 = np.array(c2, dtype=np.float32)[None, None, :]
    grad = (1.0 - t[..., None]) * c1 + t[..., None] * c2
    return grad.astype(np.uint8)

def make_radial_highkey_gradient(h, w, center=None, inner=(255, 245, 220), outer=(200, 230, 255)):
    """Bright radial gradient (center brighter)."""
    cy, cx = (h // 2, w // 2) if center is None else center
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r = (r / r.max()).astype(np.float32)  # 0 center -> 1 edge
    r = np.clip(r, 0, 1)
    inner = np.array(inner, dtype=np.float32)
    outer = np.array(outer, dtype=np.float32)
    grad = (1.0 - r)[..., None] * inner + r[..., None] * outer
    return grad.astype(np.uint8)

def make_person_hsv_layer(shape_hw, is_good, score):
    """
    Person layer in HSV:
    - Hue: warm for Good, cool for Bad
    - Value (brightness): scaled by score (0..100)
    - Subtle vertical brightness gradient
    """
    h, w = shape_hw
    hue_good = 5      # reddish warm (0..179)
    hue_bad  = 120    # cyan-blue
    H = (hue_good if is_good else hue_bad)
    S = 220
    V_base = int(np.clip(130 + (score / 100.0) * (255 - 130), 0, 255))
    v_row = np.linspace(-20, 20, h, dtype=np.float32)[:, None]
    V = np.clip(V_base + v_row, 0, 255).astype(np.uint8)

    H_img = np.full((h, w), H, dtype=np.uint8)
    S_img = np.full((h, w), S, dtype=np.uint8)
    V_img = np.repeat(V, w, axis=1).astype(np.uint8)

    hsv = cv2.merge([H_img, S_img, V_img])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def render_overlay(frame: np.ndarray, alpha: np.ndarray, is_good: bool, face_result, score_smooth: float):
    """
    Bright gradient background + score-driven bright person layer (HSV).
    Alpha composite: out = alpha*person + (1-alpha)*background
    """
    h, w = frame.shape[:2]

    # Background gradient (choose one)
    bg = make_radial_highkey_gradient(h, w, inner=(255, 250, 230), outer=(190, 225, 255))
    # bg = make_linear_gradient(h, w, c1=(255,220,180), c2=(180,220,255))

    # Person layer (brightness tracks score)
    person = make_person_hsv_layer((h, w), is_good=is_good, score=score_smooth)

    a = alpha[..., None].astype(np.float32)
    comp = (a * person.astype(np.float32) + (1.0 - a) * bg.astype(np.float32)).astype(np.uint8)

    # Facial keypoints (white for bad, red for good)
    col = (255, 255, 255) if not is_good else (0, 0, 255)
    comp = draw_keypoints_from_landmarker(comp, face_result, col)
    return comp

def annotate_hud(img: np.ndarray, lbls, score: float, is_good: bool, fps: float = None):
    color = (0, 255, 0) if is_good else (0, 0, 255)
    cv2.putText(img, f"Goodness: {score:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    y = 60
    for t in lbls:
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        y += 30
    if fps is not None:
        cv2.putText(img, f"FPS: {fps:.1f}", (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    return img

# ---------- Main ----------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="Camera index (0 or 1...)")
    ap.add_argument("--thresh_on", type=float, default=GOOD_ON, help="Good-on threshold")
    ap.add_argument("--thresh_off", type=float, default=GOOD_OFF, help="Good-off threshold")
    ap.add_argument("--ema", type=float, default=EMA_ALPHA, help="EMA smoothing factor (0..1)")
    args = ap.parse_args()

    device = pick_device()

    # Ensure deps
    ensure_package("cv2", "opencv-python")
    ensure_package("numpy")
    ensure_package("mediapipe")

    # Init Face Landmarker (VIDEO mode, blendshapes)
    landmarker = init_face_landmarker(num_faces=1)
    print("[INFO] MediaPipe Face Landmarker initialized (blendshapes enabled).")

    # Backend: try RVM; fallback to convex-hull matte
    try:
        backend = RVMBackend(device=device)
        print("[INFO] Using backend: RVM")
    except Exception as e:
        print(f"[WARN] RVM unavailable ({e}). Will use Landmarker convex-hull matte.")
        backend = None

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera (index={args.cam}).")

    ema_fps = None
    t0 = time.time()
    state = GoodBadState(on=args.thresh_on, off=args.thresh_off, alpha=args.ema)

    print("[INFO] Running. Press 'q' or ESC to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Face Landmarker: detect_for_video requires microsecond timestamp
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts_us = int(time.time() * 1e6)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        face_result = landmarker.detect_for_video(mp_image, ts_us)

        # Person matte
        if backend is not None:
            try:
                alpha = backend.infer_alpha(frame)
            except Exception:
                h, w = frame.shape[:2]
                alpha = alpha_from_landmarks(face_result, h, w)
        else:
            h, w = frame.shape[:2]
            alpha = alpha_from_landmarks(face_result, h, w)

        # Score from blendshapes (first face only)
        score_raw = 0.0
        if face_result.face_blendshapes and len(face_result.face_blendshapes) > 0:
            score_raw = compute_goodness_from_blendshapes(face_result.face_blendshapes[0])

        # Temporal smoothing + hysteresis
        score_smooth, is_good = state.update(score_raw)
        labels = labels_from_state(is_good)

        # Compose & HUD (NOTE: 5-arg render_overlay)
        out = render_overlay(frame, alpha, is_good, face_result, score_smooth)
        t1 = time.time()
        fps = 1.0 / max(1e-6, t1 - t0)
        t0 = t1
        ema_fps = fps if ema_fps is None else (0.9 * ema_fps + 0.1 * fps)
        out = annotate_hud(out, labels, score_smooth, is_good, fps=ema_fps)

        cv2.imshow("Good/Evil Overlay (RVM -> MediaPipe Tasks fallback)", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()
