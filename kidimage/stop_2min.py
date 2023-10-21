import cv2
import torch
import torchvision
import numpy as np
import time

def apply_mosaic(img, ratio=0.05):
    h, w, _ = img.shape
    small = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

# Initialize the DeepLab model
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()
model.cuda()

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize variables
score = 0
game_duration = 120
start_time = time.time()
background_frame = None
last_activity_level = 0

cv2.namedWindow("Human Coordinate Game", cv2.WINDOW_NORMAL)
cv2.imshow("Human Coordinate Game", np.zeros((400, 400, 3), dtype=np.uint8))
cv2.waitKey(1)

print("Please frame out for 5 seconds...")
time.sleep(5)
ret, background_frame = cap.read()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Prepare the frame for DeepLab model
    small_frame = cv2.resize(frame, (300, 300))
    input_array = np.array(small_frame).transpose((2, 0, 1))
    input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).cuda()

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

    # Calculate score
    activity_level = np.sum(output_predictions == 15)
    if activity_level > 100:
        score += 10
    else:
        score -= 10

    # Check game conditions
    elapsed_time = time.time() - start_time
    if elapsed_time > game_duration:
        if score >= 1000:
            game_status = "You win!"
        else:
            game_status = "You lose!"
        break

    # Apply mosaic to frame
    mosaic_frame = apply_mosaic(frame)
    
    # Create transparency effect
    alpha = 0.5
    transparent_frame = cv2.addWeighted(mosaic_frame, alpha, background_frame, 1 - alpha, 0)
    
    # Display the score
    cv2.putText(transparent_frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow("Human Coordinate Game", transparent_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        game_status = "Game over!"
        break

# Show the game status for 3 seconds
cv2.putText(frame, game_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Human Coordinate Game", frame)
cv2.waitKey(3000)

cap.release()
cv2.destroyAllWindows()
