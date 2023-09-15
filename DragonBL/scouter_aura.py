
import cv2
import numpy as np
import time

# Initialize camera and window
cap = cv2.VideoCapture(0)
cv2.namedWindow("Scouter Frame", cv2.WINDOW_AUTOSIZE)

# Initialize first_frame and power_levels list
first_frame = None
power_levels = []
max_power_level = 0

# Initialize aura_effect
aura_effect = np.zeros((480, 640, 3), dtype=np.uint8)

# Start time for the initial 5 seconds
start_time = time.time()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Initialize first_frame
    if first_frame is None:
        first_frame = gray
        continue

    # Compute absolute difference between current frame and first frame
    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Compute power level
    white_pixels = np.sum(thresh) // 255
    power_level = (white_pixels ** 1.5) // 5000
    power_levels.append(power_level)

    # Compute max power level, but initialize to 0 for the first 5 seconds
    if time.time() - start_time < 5:
        max_power_level = 0
    else:
        max_power_level = max(max_power_level, power_level)

    # Generate aura effect
    aura_effect.fill(0)
    aura_color = (0, int(max_power_level), int(max_power_level))
    aura_effect[thresh == 255] = aura_color
    aura_effect = cv2.GaussianBlur(aura_effect, (99, 99), 0)
    
    # Combine original frame with aura
    combined_frame = cv2.addWeighted(frame, 1, aura_effect, 0.5, 0)

    # Display power and max power on the frame
    cv2.putText(combined_frame, f"Power Level: {int(power_level)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(combined_frame, f"Max Power: {int(max_power_level)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show the frame
    cv2.imshow("Scouter Frame", combined_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
