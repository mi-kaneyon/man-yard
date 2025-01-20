import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
import argparse

# Custom dataset for webcam or single image
class ImageDataset(Dataset):
    def __init__(self, resize_shape=(640, 480), use_camera=True, image=None):
        """
        If use_camera=True, open a webcam.
        If use_camera=False, use the single 'image' instead.
        """
        self.resize_shape = resize_shape
        self.use_camera = use_camera
        self.cap = None
        self.image = image

        if self.use_camera:
            # Attempt to open the webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise IOError("Cannot open webcam.")

    def __len__(self):
        # For a webcam, there's no finite length, so return a large number
        if self.use_camera:
            return 999999
        else:
            # If just a single image, length is 1
            return 1

    def __getitem__(self, idx):
        if self.use_camera:
            if self.cap is None:
                return None
            ret, frame = self.cap.read()
            if not ret:
                return None  # or raise an exception
            frame = cv2.resize(frame, self.resize_shape)
        else:
            # Single image mode
            frame = cv2.resize(self.image, self.resize_shape)

        # Convert to RGB and Torch tensor
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return frame_t

    def release(self):
        if self.cap is not None:
            self.cap.release()

# Simple color reducer model
class ColorReducer(nn.Module):
    def __init__(self, n_colors):
        super(ColorReducer, self).__init__()
        self.n_colors = n_colors
        # Create a random palette on GPU
        self.palette = nn.Parameter(torch.rand(n_colors, 3).cuda())

    def forward(self, x):
        """
        x shape: (B, 3, H, W)
        Return shape: same, but each pixel is replaced by nearest color in palette
        """
        B, C, H, W = x.shape
        # Flatten to (B, H*W, 3)
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, 3)
        # Distances to palette
        distances = torch.cdist(x_flat, self.palette)  # (B, H*W, n_colors)
        labels = torch.argmin(distances, dim=2)        # (B, H*W)
        # Map each pixel
        reduced = self.palette[labels].view(B, H, W, 3).permute(0, 3, 1, 2)
        return reduced

# K-means-based palette extraction (optional usage in your code)
def get_palette_from_image(frame, n_colors=16):
    """
    Example function that extracts a palette from a single image using OpenCV k-means.
    """
    # downsize for speed
    frame_small = cv2.resize(frame, (160, 120))
    h, w, c = frame_small.shape
    data = frame_small.reshape(h * w, c).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, _, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return centers.astype(np.uint8)

# Pixelation function
def pixelate(image_bgr: np.ndarray, scale=0.1) -> np.ndarray:
    """
    Scales down by 'scale', then back up to produce blocky pixels
    """
    h, w = image_bgr.shape[:2]
    small = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated

def main():
    parser = argparse.ArgumentParser(description="Apply retro effects to image or webcam.")
    parser.add_argument("--image", type=str, help="Path to input image.")
    parser.add_argument("--width", type=int, default=640, help="Resize width")
    parser.add_argument("--height", type=int, default=480, help="Resize height")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create model
    n_colors = 16
    model = ColorReducer(n_colors).to(device)

    # Decide if using camera or single image
    use_camera = (args.image is None)
    if not use_camera:
        image = cv2.imread(args.image)
        if image is None:
            print("Error: Could not open image:", args.image)
            return
        dataset = ImageDataset(resize_shape=(args.width, args.height), use_camera=False, image=image)
    else:
        dataset = ImageDataset(resize_shape=(args.width, args.height), use_camera=True)

    # Create DataLoader with num_workers=0 for webcam
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Optional: compile model for GPU
    if device.type == 'cuda':
        try:
            model = torch.compile(model)
        except:
            pass  # If older PyTorch, compile() might not be available

    try:
        with torch.no_grad():
            for frame in dataloader:
                if frame is None or len(frame) == 0:
                    # If the dataset returned None or empty
                    break

                frame = frame.to(device)
                # Forward pass
                reduced_frame = model(frame)  # shape: (1, 3, H, W)
                # Convert to NumPy
                out = (reduced_frame * 255).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
                # Convert RGB->BGR
                out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

                # Pixelate
                retro_bgr = pixelate(out_bgr, scale=0.1)

                if not use_camera:
                    # If single image, just display once
                    cv2.imshow("Retro Image", retro_bgr)
                    cv2.waitKey(0)
                    break
                else:
                    # Webcam mode
                    cv2.imshow("Retro Webcam - 16 Colors + Pixelation", retro_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    except Exception as e:
        print("Error encountered:", e)

    finally:
        dataset.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")

if __name__ == "__main__":
    main()
