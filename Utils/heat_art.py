import cv2

# Initialize the camera with resolution 1280x720
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    # Capture frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Generate heatmap
    heatmap_color = cv2.applyColorMap(edges, cv2.COLORMAP_JET)

    # Display the resulting frame
    cv2.imshow('Heatmap', heatmap_color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
