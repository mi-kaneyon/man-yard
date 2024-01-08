import torch
import torchvision.transforms as T
import torchvision.models.detection
import cv2
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import numpy as np
import time

# GPU using setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights).to(device)
model.eval()

def apply_heatmap_effect(image):
    """ heatmap style image """
    heatmap_img = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap_img, 0.7, image, 0.3, 0)


# dictionary defenition of person
human_timers = {}

def apply_blur_to_mask(image, mask, ksize=51):
    """ blur  """
    mask = mask.mul(255).byte().cpu().numpy()
    blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    image[mask > 128] = blurred_image[mask > 128]
    return image


def apply_segmentation(frame, current_time):
    transform = T.Compose([T.ToTensor()])
    image = transform(frame).to(device).unsqueeze(0)
    with torch.no_grad():
        prediction = model(image)

    for element in range(len(prediction[0]['masks'])):
        score = prediction[0]['scores'][element].item()
        label = prediction[0]['labels'][element].item()
        mask = prediction[0]['masks'][element, 0]

        if score > 0.5:
            if label == 1:  # person
                frame = apply_blur_to_mask(frame, mask)
            elif label == 17:  # cat
                mask_cpu = mask.cpu().numpy()  # from GPU to CPU
                frame[mask_cpu > 0.5] = [0, 255, 0]  # Green

    return frame

# camera device
cap1 = cv2.VideoCapture(0)  # ID cam1
cap2 = cv2.VideoCapture(2)  # ID cam2

while True:
    # confirm current time
    current_time = time.time()	
    # get realtime image from cam
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # application of heatmap
    heatmap_frame1 = apply_heatmap_effect(frame1)
    heatmap_frame2 = apply_heatmap_effect(frame2)

    # Segmentation with current time
    segmented_frame1 = apply_segmentation(heatmap_frame1, current_time)
    segmented_frame2 = apply_segmentation(heatmap_frame2, current_time)


    # combine 2cam image
    blended_frame = cv2.addWeighted(segmented_frame1, 0.5, segmented_frame2, 0.5, 0)

    # result display
    cv2.imshow('Blended Segmented Camera', blended_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
