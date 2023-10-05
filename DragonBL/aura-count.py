import cv2
import numpy as np
import time
from torchvision.models.segmentation import deeplabv3_resnet101
import torch

# Initialize the DeepLab model
model = deeplabv3_resnet101(pretrained=True).eval()

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster inference
    small_frame = cv2.resize(frame, (300, 300))

    # Prepare the frame for DeepLab model
    input_array = np.array(small_frame).transpose((2, 0, 1))
    input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        start_time = time.time()
        output = model(input_tensor)['out'][0]
        end_time = time.time()
        inference_time = end_time - start_time

    # Get the segmentation mask
    output_predictions = output.argmax(0).byte().cpu().numpy()

    # Generate heatmap based on the score
    heatmap_color = np.zeros((output_predictions.shape[0], output_predictions.shape[1], 3), dtype=np.uint8)
    heatmap_color[output_predictions == 15] = [0, 0, 255]  # Red for human

    # Calculate score directly from the activity level
    activity_level = np.sum(output_predictions == 15)
    score = activity_level // 100  # Just an example, you can adjust the conversion factor

    # Map the color based on the score
    if score < 25:
        color = (0, 0, 255)  # Red
    elif score < 50:
        color = (0, 255, 255)  # Yellow
    elif score < 75:
        color = (0, 255, 0)  # Green
    else:
        color = (255, 0, 0)  # Blue

    # Prepare the display frame
    display_frame = cv2.resize(heatmap_color, (frame.shape[1], frame.shape[0]))
    cv2.putText(display_frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    print(f"Inference time: {inference_time:.3f} seconds")
    print(f"Activity Level: {activity_level}")
    print(f"Score: {score}")

    # Show the frame
    cv2.imshow("Activity Monitor", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
