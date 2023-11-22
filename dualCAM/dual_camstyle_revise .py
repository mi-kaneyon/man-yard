import cv2
import numpy as np

# Initialize the cameras
cap1 = cv2.VideoCapture(0)  # Left camera
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

cap2 = cv2.VideoCapture(2)  # Right camera
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# Initialize StereoBM matcher with adjusted parameters
stereo = cv2.StereoBM_create(numDisparities=96, blockSize=21)  # Adjust these parameters as needed

# Initialize WLS filter with adjusted parameters
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(80000)  # Increased for smoother disparities
wls_filter.setSigmaColor(2.0)  # Adjust for color-based filtering

while True:
    # Capture frames from both cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # Check if frames are captured successfully
    if not (ret1 and ret2):
        print("Failed to grab frames")
        break

    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute disparity using StereoBM
    disparity_left = stereo.compute(gray1, gray2)
    disparity_left = cv2.normalize(disparity_left, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply WLS filter
    filtered_disparity = wls_filter.filter(disparity_left, gray1, None, gray2)

    # Normalize for visualization
    filtered_disparity = cv2.normalize(src=filtered_disparity, dst=filtered_disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filtered_disparity = np.uint8(filtered_disparity)

    # Generate heat map
    heat_map = cv2.applyColorMap(filtered_disparity, cv2.COLORMAP_JET)

    # Show the heat map
    cv2.imshow('Filtered Disparity Map', heat_map)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the cameras and destroy windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
