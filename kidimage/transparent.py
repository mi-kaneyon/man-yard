import cv2
import torch
import numpy as np
import time
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101

# Initialize the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained model
model = deeplabv3_resnet101(pretrained=True).eval().to(device)

# Initialize camera
cap = cv2.VideoCapture(0)

# Wait for 5 seconds to capture the background
print("Capturing background in 5 seconds...")
time.sleep(5)
ret, background_frame = cap.read()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    # Prepare the frame for the model
    input_frame = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(frame).unsqueeze(0).to(device)

    # Make predictions
    with torch.no_grad():
        output = model(input_frame)['out'][0]
    output_predictions = torch.argmax(output, dim=0).byte().cpu().numpy()

    # Resize the output to match the frame size
    output_predictions_resized = cv2.resize(output_predictions, (frame.shape[1], frame.shape[0]))

# Replace the human part with the background
    human_mask = output_predictions_resized == 15
    replaced_frame = np.where(human_mask[:,:,None], background_frame, frame)


    # Show the frame
    cv2.imshow('Frame', replaced_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()
