import cv2
import numpy as np
import random

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize the happiness score
happiness_score = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Apply a mosaic filter to the entire body (for demonstration purposes)
    frame_small = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
    frame_mosaic = cv2.resize(frame_small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Dummy logic to update the happiness score based on body movements
    # Note: In a real application, you would use a more sophisticated model to evaluate the body pose
    is_happy_movement = random.choice([True, False])
    
    if is_happy_movement:
        happiness_score = min(happiness_score + 10, 100)
        color = (255, 105, 180)  # Pink
    else:
        happiness_score = max(happiness_score - 10, -100)
        color = (0, 0, 0)  # Black
    
    cv2.putText(frame_mosaic, f"Happiness Score: {happiness_score}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('Are you happy?', frame_mosaic)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
