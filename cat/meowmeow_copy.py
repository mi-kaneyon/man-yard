import cv2
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

# Initialize the DeepLab model and move it to CUDA
model = models.segmentation.deeplabv3_resnet101(pretrained=True).to('cuda')
model.eval()

# Initialize camera and set FPS
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 20)

# Initialize variables
num_clones = 0
alpha = 0.5  # Transparency level

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for model
    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).to('cuda')

    # Run the frame through the model
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = torch.argmax(output, dim=0).byte().cpu().numpy()

    # Create mask and clones
    mask = (output_predictions == 8)
    masked_frame = np.zeros_like(frame)
    masked_frame[mask] = frame[mask]

    final_frame = np.copy(frame)
    h, w, _ = frame.shape
    offsets = np.linspace(-w // 4, w // 4, num_clones)

    for offset in offsets:
        translation_matrix = np.float32([[1, 0, offset], [0, 1, 0]])
        clone = cv2.warpAffine(masked_frame, translation_matrix, (w, h))

        mask_clone = clone > 0

        # Check if the mask_clone has any True values before proceeding
        if np.any(mask_clone):
            rows, cols, _ = np.where(mask_clone)
            for r, c in zip(rows, cols):
                temp_final = final_frame[r, c][np.newaxis, :]
                temp_clone = clone[r, c][np.newaxis, :]
                final_frame[r, c] = cv2.addWeighted(temp_final, 1 - alpha, temp_clone, alpha, 0).flatten()

    # Display the resulting frame
    cv2.imshow('Cat Cloner', final_frame)

    key = cv2.waitKey(1)

    # Increase or decrease the number of clones
    if key == ord('A') or key == ord('a'):
        num_clones = min(num_clones + 1, 5)
    elif key == ord('D') or key == ord('d'):
        num_clones = max(num_clones - 1, 0)
    elif key == ord('Q') or key == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
