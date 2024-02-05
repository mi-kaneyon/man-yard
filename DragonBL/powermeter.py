import cv2
import numpy as np

cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2()

def detect_motion_and_draw_box(cap, backSub):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fgMask = backSub.apply(frame)
        heatmap = cv2.applyColorMap(fgMask, cv2.COLORMAP_JET)
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                color = (0, 255, 0) if cv2.contourArea(contour) < 500 else (0, 0, 255)
                cv2.rectangle(heatmap, (x, y), (x+w, y+h), color, 2)
                score = cv2.contourArea(contour)
                cv2.putText(heatmap, f"Score: {score}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow('Frame', heatmap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

detect_motion_and_draw_box(cap, backSub)
cap.release()
cv2.destroyAllWindows()
