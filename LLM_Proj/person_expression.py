import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from transformers import AutoTokenizer, AutoModelForCausalLM
import cv2
import os
from datetime import datetime
import time
import gc

def capture_image_with_human():
    print("デバイスとモデルを初期化します。")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    model.eval()

    print("カメラを開始します。")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが開けません。カメラ設定を確認してください。")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_save_path = "detected_images"
    os.makedirs(image_save_path, exist_ok=True)
    detected_image_path = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("カメラからフレームを読み取ることができません。")
                continue

            image = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(image)

            human_detected = False
            for box, score, label in zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels']):
                if score >= 0.3 and label == 1:  # スコアの閾値を0.5から0.3に下げる
                    box = box.detach().cpu().numpy().astype(int)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    human_detected = True

            if human_detected:
                cv2.imshow('Detected Persons', frame)
                time.sleep(1)  # 検出間隔を2秒から1秒に短縮
                ret, frame = cap.read()  # 再度フレームを取得
                if not ret:
                    print("カメラからフレームを読み取ることができません。")
                    continue

                image = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(image)

                # 人間がフレームにしっかり入っているか再確認
                for box, score, label in zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels']):
                    if score >= 0.3 and label == 1:  # スコアの閾値を0.5から0.3に下げる
                        box = box.detach().cpu().numpy().astype(int)
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        img_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
                        img_path = os.path.join(image_save_path, img_name)
                        cv2.imwrite(img_path, frame)  # フルフレームを保存
                        detected_image_path = img_path
                        print(f"画像保存: {img_path}")
                        break

                if detected_image_path:
                    break

            cv2.imshow('Live', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("ユーザーにより終了されました。")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # モデルのメモリを解放
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return detected_image_path

def process_image_with_llm(image_path):
    model_path = "Rakuten/RakutenAI-7B-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    model.eval()

    questions_japanese = [
        "この画像に写っている人物の性格について説明してください。",
        "この画像に写っている人物の好きそうなものについて説明してください。",
        "この画像に写っている人物の名前について説明してください。"
    ]

    japanese_descriptions = []

    for question_japanese in questions_japanese:
        inputs = tokenizer(question_japanese, return_tensors="pt")
        inputs = inputs.to("cuda")
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=150,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,  # サンプリングを有効にしてランダム性を導入
                top_k=50,  # トップKサンプリング
                top_p=0.95  # トップPサンプリング
            )
        japanese_description = tokenizer.decode(output[0], skip_special_tokens=True)
        japanese_descriptions.append(f"質問: {question_japanese}\n回答: {japanese_description}\n")
        print(f"質問: {question_japanese}")
        print(f"回答: {japanese_description}")

    # 日本語の説明をテキストファイルに保存
    with open(image_path.replace(".png", "_description.txt"), "w") as file:
        for line in japanese_descriptions:
            file.write(line)

    # 画像に説明をオーバーレイ
    img = cv2.imread(image_path)
    overlay_text = "Please check your attached text for person's images."
    y0, dy = 30, 30
    cv2.putText(img, overlay_text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # 文字色を赤に変更

    overlay_img_path = image_path.replace(".png", "_with_description.png")
    cv2.imwrite(overlay_img_path, img)
    print(f"説明付き画像保存: {overlay_img_path}")

    # モデルのメモリを解放
    del model
    torch.cuda.empty_cache()
    gc.collect()

detected_image_path = capture_image_with_human()
if detected_image_path:
    process_image_with_llm(detected_image_path)
else:
    print("No images detected.")
