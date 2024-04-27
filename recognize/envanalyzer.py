import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import json

# ImageNetのクラスインデックスをロード
with open('imagenet_class_index.json') as f:
    class_idx = json.load(f)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

def initialize_model():
    # ResNet50モデルの初期化
    model = models.resnet50(pretrained=True)
    model.eval()
    return model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

def preprocess_image(image):
    # 画像の前処理
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    return image.unsqueeze(0)

def predict_image(model, image):
    # 画像から予測する
    with torch.no_grad():
        outputs = model(image)
    return outputs

def main():
    model = initialize_model()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera is not available.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to get the frame.")
            continue

        image = Image.fromarray(frame)
        image_tensor = preprocess_image(image).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        outputs = predict_image(model, image_tensor)
        percentages = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        
        # トップ5予測結果を取得
        _, top5 = torch.topk(percentages, 5)
        result_strs = [f"{idx2label[idx]}: {percentages[idx].item():.2f}%" for idx in top5]

        # 結果を縦に列挙して表示
        y_position = 15
        for s in result_strs:
            cv2.putText(frame, s, (5, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_position += 15

        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
