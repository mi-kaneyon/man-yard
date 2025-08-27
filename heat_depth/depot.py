import os # Added to suppress TF-TRT Warning
import cv2
import torch
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor # NEW IMPORTS

# --- Suppress TF-TRT Warning ---
# This line sets the logging level for TensorFlow to suppress warnings and info messages.
# This is useful if TensorFlow is installed but not directly used by the PyTorch MiDaS model,
# and you want to prevent its TensorRT warnings from cluttering the console.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # '2' for warnings and info, '3' for errors only

# --- 1. Global Variables for Trackbars ---
# Default values for GUI sliders.
# Adjusting these to provide a less obtrusive initial heatmap visualization.

# Depth intensity multiplier.
# Lower value makes depth variations appear more subtle, allowing the original image to be clearer.
# Initial value set to 70 (from 100) means depth values are multiplied by 0.7 before normalization.
depth_strength = 70

# Blending alpha (transparency) for the heatmap overlay.
# Lower value makes the heatmap more transparent, revealing more of the original frame.
# Initial value set to 50 (from 70) means the heatmap is 50% transparent, and the original frame is 50% visible.
alpha_blend = 50

# Index for the currently selected OpenCV colormap.
# Default to 0, which typically corresponds to COLORMAP_JET.
colormap_idx = 0     


# A selection of OpenCV colormaps to cycle through via the 'Colormap' trackbar.
COLORMAPS = [
    cv2.COLORMAP_JET, cv2.COLORMAP_MAGMA, cv2.COLORMAP_INFERNO,
    cv2.COLORMAP_PLASMA, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_PARULA,
    cv2.COLORMAP_TURBO, cv2.COLORMAP_CIVIDIS, cv2.COLORMAP_TWILIGHT,
    cv2.COLORMAP_RAINBOW, cv2.COLORMAP_OCEAN
]
# Corresponding names for display or debugging (not used in this script, but good for reference).
COLORMAP_NAMES = [
    "JET", "MAGMA", "INFERNO", "PLASMA", "VIRIDIS", "PARULA",
    "TURBO", "CIVIDIS", "TWILIGHT", "RAINBOW", "OCEAN"
]

# --- 2. Callback Functions for Trackbars ---
# These functions are called automatically when their respective trackbar's value changes.

def on_depth_strength_change(val):
    """Updates the global depth_strength variable based on the trackbar's value."""
    global depth_strength
    depth_strength = val

def on_colormap_change(val):
    """Updates the global colormap_idx variable based on the trackbar's value."""
    global colormap_idx
    colormap_idx = val

def on_alpha_blend_change(val):
    """Updates the global alpha_blend variable based on the trackbar's value."""
    global alpha_blend
    alpha_blend = val

# --- 3. Initialize PyTorch MiDaS Model ---
print("Loading MiDaS depth estimation model...")

# Load the image processor and the DPTForDepthEstimation model from Hugging Face.
# The processor handles image resizing, normalization, and other preprocessing steps
# required by the MiDaS model, simplifying the input pipeline.
processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")

# Determine the device to run the model on (GPU if available, otherwise CPU).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # Move the model to the selected device.
model.eval() # Set the model to evaluation mode to disable dropout, batch normalization updates, etc.

# For MiDaS, the `transform` from torchvision is largely replaced by the `processor`.
# We'll use the processor directly on the NumPy/PIL Image.
print(f"MiDaS model loaded on device: {device}")

# --- 4. Initialize Video Capture ---
# Attempt to open the default camera (index 0).
# If you have multiple cameras or the default doesn't work, you might need to try other indices
# (e.g., cv2.VideoCapture(1), cv2.VideoCapture(2), etc.) or implement a camera detection loop.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    # Exit the application if the camera cannot be opened.
    # In a production application, you might want to display a static image,
    # log the error, or provide more user-friendly feedback.
    exit()

# Retrieve the original frame dimensions from the camera.
# These dimensions will be used to resize the model's depth output back to the original size
# before blending it with the live video frame.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- 5. Create GUI Windows and Trackbars ---
# Create two OpenCV windows: one for displaying the video, and one for adjustment trackbars.
cv2.namedWindow('Real-time Depth Heatmap', cv2.WINDOW_NORMAL)
cv2.namedWindow('Adjustments', cv2.WINDOW_NORMAL)

# Create trackbars for adjusting depth strength, colormap, and blend alpha.
# Parameters: (trackbar_name, window_name, default_value, max_value, callback_function)
cv2.createTrackbar('Depth Strength', 'Adjustments', depth_strength, 200, on_depth_strength_change) # Max 2.0x multiplier
cv2.createTrackbar('Colormap', 'Adjustments', colormap_idx, len(COLORMAPS) - 1, on_colormap_change) # Cycles through available colormaps
cv2.createTrackbar('Blend Alpha', 'Adjustments', alpha_blend, 100, on_alpha_blend_change) # Max 1.0 alpha value

# --- 6. Main Loop for Real-time Processing ---
print("Starting real-time processing. Press 'q' to quit.")
while True:
    # Read a frame from the video capture.
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break # Exit the loop if frame reading fails.

    # Convert the OpenCV BGR frame to RGB format, which is typically expected by PyTorch models.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- PyTorch MiDaS Inference ---
    # Process the RGB frame using the MiDaS image processor.
    # The processor returns a dictionary containing processed pixel values as a PyTorch tensor.
    inputs = processor(images=rgb_frame, return_tensors="pt")

    # Move the input pixel values to the selected device (GPU for faster inference).
    pixel_values = inputs["pixel_values"].to(device)

    # Perform inference without calculating gradients to save memory and speed up computation.
    with torch.no_grad():
        outputs = model(pixel_values)
        # MiDaS models typically output 'predicted_depth'.
        predicted_depth = outputs.predicted_depth

    # Upscale the predicted depth map back to the original frame's dimensions.
    # `unsqueeze(1)` adds a channel dimension for `interpolate` function.
    # `bicubic` mode provides good quality resizing. `align_corners=False` is common practice.
    # `squeeze().cpu().numpy()` removes unnecessary dimensions, moves to CPU, and converts to NumPy array.
    depth_map = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(frame_height, frame_width),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    # --- Apply Depth Strength ---
    # Scale the depth map values using the current `depth_strength` from the trackbar.
    # `depth_strength` is scaled from 0-200 to a 0.0-2.0 multiplier.
    # This emphasizes or de-emphasizes the relative depth differences.
    scaled_depth_map = depth_map * (depth_strength / 100.0)

    # Normalize the scaled depth map to the 0-255 range and convert it to an 8-bit unsigned integer.
    # This is required before applying an OpenCV colormap.
    normalized_depth = cv2.normalize(scaled_depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # --- Apply Colormap ---
    # Get the currently selected colormap based on the `colormap_idx` from the trackbar.
    current_colormap = COLORMAPS[colormap_idx]
    # Apply the colormap to the normalized depth map to create a colored heatmap image.
    colored_heatmap = cv2.applyColorMap(normalized_depth, current_colormap)

    # --- Blend with Original Frame ---
    # Convert `alpha_blend` (0-100) to a float alpha value (0.0-1.0).
    alpha = alpha_blend / 100.0
    # Blend the original video frame with the colored heatmap.
    # `cv2.addWeighted` calculates `(frame * (1 - alpha)) + (colored_heatmap * alpha)`.
    # A higher `alpha` means the heatmap is more prominent.
    blended_frame = cv2.addWeighted(frame, 1 - alpha, colored_heatmap, alpha, 0)

    # --- Display Results ---
    # Show the blended frame in the main display window.
    cv2.imshow('Real-time Depth Heatmap', blended_frame)

    # --- Check for Exit Key ---
    # Wait for 1 millisecond for a key press.
    # If 'q' is pressed, break the loop to exit the application.
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# --- 7. Cleanup ---
# Release the camera resource.
cap.release()
# Close all OpenCV windows.
cv2.destroyAllWindows()
print("Application closed.")
