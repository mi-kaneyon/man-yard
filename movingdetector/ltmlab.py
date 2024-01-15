import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

torch.cuda.init()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lstm_model = SimpleLSTM(input_size=2, hidden_size=128, output_size=10).to(device)
lstm_model.eval()

resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
resnet_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

backSub = cv2.createBackgroundSubtractorMOG2()

timeseries_data = []

max_data_points = 1000

def process_frame(frame, resnet_model, lstm_model, transform, device, timeseries_data, max_data_points):
    fgMask = backSub.apply(frame)
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = frame[y:y + h, x:x + w]
            roi = transform(roi).unsqueeze(0).to(device)

            with torch.no_grad():
                resnet_output = resnet_model(roi)

            center_point = np.array([x + w / 2, y + h / 2])
            if len(timeseries_data) >= max_data_points:
                timeseries_data.pop(0)
            timeseries_data.append(center_point)

    if len(timeseries_data) > 0:
        lstm_input = torch.tensor(np.array(timeseries_data), dtype=torch.float).unsqueeze(0).to(device)
        with torch.no_grad():
            lstm_output = lstm_model(lstm_input)
            probabilities = torch.softmax(lstm_output, dim=1)
            top_p, top_class = probabilities.topk(1, dim=1)
            label = top_class.item()
            score = top_p.item()
            cv2.putText(frame, f'Label: {label}, Score: {score:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, label, score

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, label, score = process_frame(frame, resnet_model, lstm_model, transform, device, timeseries_data, max_data_points)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
