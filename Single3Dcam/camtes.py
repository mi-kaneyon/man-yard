import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

def capture_and_process_frames(camera_index, focus_start, focus_end, focus_step, brightness_factor):
    # Open the camera
    cap = cv2.VideoCapture(camera_index)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set up the display window
    display_window_name = "Pseudo-3D Point Cloud"
    cv2.namedWindow(display_window_name)
    
    # Initialize a dictionary to store depth colors for each point
    depth_map = None
    collecting = False
    collection_count = 0
    
    while True:
        if collecting and collection_count < 20:
            for focus in range(focus_start, focus_end + 1, focus_step):
                # Set the camera focus
                cap.set(cv2.CAP_PROP_FOCUS, focus)
                
                # Capture frame-by-frame
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Could not read frame.")
                    break
                
                # Adjust the brightness of the frame
                frame = adjust_brightness(frame, brightness_factor)
                
                # Process the frame and update the depth map
                depth_map = update_depth_map(frame, focus, depth_map)
                
                # Visualize the depth map
                processed_frame = visualize_depth_map(frame, depth_map)
                
                # Display the processed frame
                cv2.imshow(display_window_name, processed_frame)
                
                # Wait for a key press and break if 'q' is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord('s'):
                    collecting = not collecting
                    collection_count = 0  # Reset the collection count
                
            collection_count += 1
        else:
            # Set the focus to a fixed value after collection is done
            cap.set(cv2.CAP_PROP_FOCUS, (focus_start + focus_end) // 2)
            
            # Display the last processed frame
            if depth_map is not None:
                processed_frame = visualize_depth_map(frame, depth_map)
                cv2.imshow(display_window_name, processed_frame)
            
            # Wait for a key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                collecting = not collecting
                collection_count = 0  # Reset the collection count
    
    cap.release()
    cv2.destroyAllWindows()

# Function to adjust the brightness of a frame
def adjust_brightness(frame, factor):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frame

# Function to update the depth map based on the focus and frame using PyTorch
def update_depth_map(frame, focus, depth_map):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convert the grayscale frame to a PyTorch tensor and move to GPU
    tensor_frame = transforms.ToTensor()(gray_frame).unsqueeze(0).cuda()
    
    # Compute the Laplacian to detect edges and focus quality using PyTorch
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device='cuda').unsqueeze(0).unsqueeze(0)
    laplacian = torch.nn.functional.conv2d(tensor_frame, laplacian_kernel, padding=1).pow(2).mean().item()
    
    # Initialize the depth map if it's the first frame
    if depth_map is None:
        depth_map = np.zeros_like(frame, dtype=np.uint8)
    
    # Create a mask based on the Laplacian (focus quality)
    mask = (gray_frame > 50).astype(np.uint8)  # Thresholding for more coverage
    
    # Update the depth map with the new focus value where the mask is valid
    depth_map[mask > 0] = get_depth_color(focus)
    
    return depth_map

# Function to get a color based on the depth (focus value)
def get_depth_color(focus):
    # Normalize the focus value to a range of 0-255
    normalized_focus = int((focus / 255) * 255)
    
    # Generate a color (blue for close, red for far)
    color = (255 - normalized_focus, 0, normalized_focus)
    
    return color

# Function to visualize the depth map as points
def visualize_depth_map(frame, depth_map):
    # Create an empty canvas
    colored_frame = np.zeros_like(frame, dtype=np.uint8)
    
    # Apply the depth map colors to the canvas
    mask = np.sum(depth_map, axis=-1) > 0  # Check if the depth map has valid values
    
    points = np.column_stack(np.where(mask))  # Get the coordinates of the points
    colors = depth_map[mask]  # Get the colors of the points
    
    for point, color in zip(points, colors):
        cv2.circle(colored_frame, (point[1], point[0]), 1, tuple(int(c) for c in color), -1)
    
    return colored_frame

if __name__ == "__main__":
    camera_index = 0
    focus_start = 0
    focus_end = 255
    focus_step = 10
    brightness_factor = 2.0
    
    capture_and_process_frames(camera_index, focus_start, focus_end, focus_step, brightness_factor)
                # Visual
