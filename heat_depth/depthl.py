import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

# --- 1. Global Variables for Trackbars ---
# Default values
depth_strength = 100 # Represents a multiplier (e.g., 1.0) scaled up for trackbar
colormap_idx = 0     # Index for selected colormap
alpha_blend = 70     # Blending alpha (0-100 for 0.0-1.0)

# OpenCV Colormaps (a selection)
COLORMAPS = [
    cv2.COLORMAP_JET, cv2.COLORMAP_MAGMA, cv2.COLORMAP_INFERNO,
    cv2.COLORMAP_PLASMA, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_PARULA,
    cv2.COLORMAP_TURBO, cv2.COLORMAP_CIVIDIS, cv2.COLORMAP_TWILIGHT,
    cv2.COLORMAP_RAINBOW, cv2.COLORMAP_OCEAN
]
COLORMAP_NAMES = [
    "JET", "MAGMA", "INFERNO", "PLASMA", "VIRIDIS", "PARULA",
    "TURBO", "CIVIDIS", "TWILIGHT", "RAINBOW", "OCEAN"
]

# --- 2. Callback Functions for Trackbars ---
def on_depth_strength_change(val):
    global depth_strength
    depth_strength = val

def on_colormap_change(val):
    global colormap_idx
    colormap_idx = val

def on_alpha_blend_change(val):
    global alpha_blend
    alpha_blend = val

# --- 3. Initialize PyTorch Model ---
print("Loading PyTorch depth estimation model...")
# Example: Using a simplified MiDaS model
# You might need to adjust this based on the specific MiDaS variant or your custom model
# For a more robust MiDaS, consider using the Hugging Face transformers library
# from transformers import DPTForDepthEstimation, DPTImageProcessor
# processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
# model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")

# --- PLACEHOLDER FOR YOUR ACTUAL PYTORCH MODEL LOAD ---
# For demonstration, let's create a dummy model and transform
class DummyDepthModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # A very basic conv layer to simulate output
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1)
    def forward(self, x):
        # Simulate depth output (single channel)
        return torch.sigmoid(self.conv(x))

model = DummyDepthModel() # Replace with your actual model
# --- END PLACEHOLDER ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # Set model to evaluation mode

# Define the transformation pipeline for your model
# This should match what your PyTorch model expects
transform = transforms.Compose([
    transforms.ToPILImage(), # Convert NumPy array to PIL Image
    transforms.Resize((256, 256)), # Example size, adjust as needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Example normalization
])
print(f"PyTorch model loaded on device: {device}")

# --- 4. Initialize Video Capture ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Get original frame dimensions for resizing model output back
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- 5. Create GUI Windows and Trackbars ---
cv2.namedWindow('Real-time Depth Heatmap', cv2.WINDOW_NORMAL)
cv2.namedWindow('Adjustments', cv2.WINDOW_NORMAL)

cv2.createTrackbar('Depth Strength', 'Adjustments', depth_strength, 200, on_depth_strength_change) # Max 2.0x
cv2.createTrackbar('Colormap', 'Adjustments', colormap_idx, len(COLORMAPS) - 1, on_colormap_change)
cv2.createTrackbar('Blend Alpha', 'Adjustments', alpha_blend, 100, on_alpha_blend_change) # Max 1.0

# --- 6. Main Loop for Real-time Processing ---
print("Starting real-time processing. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to RGB for PyTorch (OpenCV reads BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- PyTorch Inference ---
    input_tensor = transform(rgb_frame).unsqueeze(0).to(device) # Add batch dimension, move to device

    with torch.no_grad():
        # model_output will be your depth map or similar
        # For a DummyDepthModel, it's a 1-channel tensor
        # For MiDaS, it's model(input_tensor).predicted_depth
        model_output = model(input_tensor)

    # Move output back to CPU and convert to NumPy
    # Detach from graph, move to CPU, convert to NumPy
    depth_map = model_output.squeeze().cpu().numpy() # Remove batch and channel dims

    # --- Apply Depth Strength ---
    # Scale depth map for visualization. `depth_strength` is 0-200 (for 0.0-2.0 multiplier)
    scaled_depth_map = depth_map * (depth_strength / 100.0)

    # Normalize depth map to 0-255 for colormap application
    # Ensure it's 8-bit unsigned integer
    normalized_depth = cv2.normalize(scaled_depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # --- Apply Colormap ---
    current_colormap = COLORMAPS[colormap_idx]
    colored_heatmap = cv2.applyColorMap(normalized_depth, current_colormap)

    # Resize heatmap to original frame dimensions
    colored_heatmap_resized = cv2.resize(colored_heatmap, (frame_width, frame_height))

    # --- Blend with Original Frame ---
    # `alpha_blend` is 0-100 (for 0.0-1.0 alpha value)
    alpha = alpha_blend / 100.0
    blended_frame = cv2.addWeighted(frame, 1 - alpha, colored_heatmap_resized, alpha, 0)
    # The `frame` (original video) and `colored_heatmap_resized` (overlay) must have the same size and type.

    # --- Display Results ---
    cv2.imshow('Real-time Depth Heatmap', blended_frame)

    # --- Check for Exit Key ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# --- 7. Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Application closed.")
