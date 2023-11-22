import cv2
import numpy as np

# Open two cameras
cap0 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

# Check if camera opened successfully
if not cap0.isOpened() or not cap2.isOpened():
    print("Error: Camera not accessible")
    exit()

cv2.namedWindow('Enhanced Heatmap', cv2.WINDOW_AUTOSIZE)

while True:
    # Capture frame-by-frame from each camera
    ret0, frame0 = cap0.read()
    ret2, frame2 = cap2.read()

    if not ret0 or not ret2:
        # Break the loop if frames are not received properly
        print("Error: Unable to capture video")
        break

    # Combine the images (for demonstration, we're simply averaging the two images)
    combined_frame = cv2.addWeighted(frame0, 0.5, frame2, 0.5, 0)

    # Here you would have your depth data which should be normalized from 0 to 255
    # For demonstration, we use a grayscale conversion as a placeholder for actual depth data
    depth_data = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(depth_data, cv2.COLORMAP_JET)

    # Increase the heatmap intensity for closer objects by reducing their corresponding alpha values
    alpha_mask = 1 - (depth_data / 255.0) ** 2  # Squaring to reduce transparency for closer objects more
    alpha_mask = cv2.merge([alpha_mask, alpha_mask, alpha_mask])

    # Blend the heatmap with the original image using the alpha mask
    blended_heatmap = (alpha_mask * combined_frame.astype(np.float32) + (1 - alpha_mask) * heatmap.astype(np.float32)).astype(np.uint8)

    # Display the resulting frame
    cv2.imshow('Enhanced Heatmap', blended_heatmap)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap0.release()
cap2.release()
cv2.destroyAllWindows()
