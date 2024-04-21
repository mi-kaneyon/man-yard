import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import spacy
import gc
import random  # random モジュールをインポート

# 日本語NLPモデルのロード
nlp = spacy.load("ja_core_news_sm")

def extract_nouns(text):
    """テキストから名詞を抽出してリストで返す関数"""
    doc = nlp(text)
    nouns = set(token.text for token in doc if token.pos_ == 'NOUN')  # 重複を避けるためにsetを使用
    return list(nouns)

model_path = "Rakuten/RakutenAI-7B-chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
model.eval()

if torch.cuda.is_available():
    model.to("cuda")

def generate_response(input_text, history=[]):
    with torch.no_grad():
        prompt_text = " | ".join(history + [input_text])
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to("cuda")
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response

def clear_memory():
    if torch.cuda.is_available():
        model.to('cpu')
        torch.cuda.empty_cache()
    gc.collect()

print("チャットボットを開始します。お題を入力してください:")
user_input = input("USER: ")
conversation_log = ["USER: " + user_input]
response = generate_response(user_input)
conversation_log.append("AI: " + response)

def create_specific_question(nouns):
    if not nouns:
        return "現在の話題についてどう思いますか？"
    topic = random.choice(nouns)  # ランダムに名詞を一つ選ぶ
    return f"{topic}について詳しく教えてください。"

# 会話セッションの改善
try:
    for _ in range(2):
        important_nouns = extract_nouns(response)
        new_topic = "次の話題: " + ", ".join(important_nouns)
        new_question = create_specific_question(important_nouns)
        conversation_log.append("AI: " + new_question)
        response = generate_response(new_question, [new_question])
        conversation_log.append("AI: " + response)
finally:
    clear_memory()

print("\n会話ログ:")
for entry in conversation_log:
    print(entry)

print("\nサマリー:")
print("この会話では以下のトピックについて議論しました：" + ", ".join(extract_nouns(conversation_log[-1])))
