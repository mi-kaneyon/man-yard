import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import torchvision.models as models

# 仮定したモデルアーキテクチャ（ResNet18）を定義します。
# 実際には訓練されたモデルのアーキテクチャに合わせて変更する必要があります。
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)  # 4は出力クラス数を仮定しています

# 学習済みの重みをロード
model.load_state_dict(torch.load('script/distance_model.pth'))
model.eval()

# 画像変換用のトランスフォーム
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 推論結果から距離を取得する関数
def output_to_distance(output):
    # この関数はモデルの出力形式に基づいて実装する必要があります。
    # ここでは、出力が距離クラスの確率であると仮定して、最も高い確率を持つクラスを距離として返します。
    _, predicted_class = torch.max(output, 1)
    # ここで predicted_class を実際の距離値にマッピングします。
    distance_labels = ['50cm', '1m', '1.5m', '2m']
    distance = distance_labels[predicted_class]
    return distance

# カメラデバイスのセットアップ
cap = cv2.VideoCapture(0)

while True:
    # 画像をキャプチャ
    ret, frame = cap.read()
    if not ret:
        break
    
    # 画像をモデルが受け入れる形に変換
    input_tensor = transform(frame)
    input_batch = input_tensor.unsqueeze(0)

    # 推論を実行
    with torch.no_grad():
        output = model(input_batch)

    # 推論結果から距離を取得
    distance = output_to_distance(output)

    # 距離をフレームにオーバーレイ
    cv2.putText(frame, f'Distance: {distance}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ヒートマップを生成してフレームにオーバーレイ
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(grayscale, cv2.COLORMAP_JET)
    frame_with_heatmap = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

    # 結果を表示
    cv2.imshow('frame', frame_with_heatmap)

    # 'q'を押すとループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
