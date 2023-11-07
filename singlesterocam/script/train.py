import torch
import torchvision
from torchvision import datasets, models, transforms
import os

# データセットのパス
data_dir = './dataset'

# モデルを保存するパス
model_save_path = './script/distance_model.pth'

# データの前処理定義
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 訓練データセットの読み込み
train_dataset = datasets.ImageFolder(data_dir, data_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# モデルの定義
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(train_dataset.classes))  # 出力層のクラス数を変更
model = model.to(device)

# 損失関数と最適化手法の定義
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 訓練ループ
for epoch in range(25):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

# モデルの保存
torch.save(model.state_dict(), model_save_path)
