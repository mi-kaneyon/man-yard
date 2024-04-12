import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "Rakuten/RakutenAI-7B-chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

# CUDAが利用可能か確認し、可能ならばモデルをGPUに、不可能ならばCPUに設定
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

print("チャットボットを開始します。何か質問はありますか？(終了するには 'exit' と入力してください)")
while True:
    user_input = input("USER: ")
    if user_input.lower() == "exit":
        break

    input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)

    # 推論の実行
    output_ids = model.generate(input_ids, max_length=256, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"ASSISTANT: {response}")
