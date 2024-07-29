import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import random
import pandas as pd
from datetime import datetime
import gc
import re

# モデルの読み込み
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    return tokenizer, model, streamer

# GPUメモリを解放する関数
def free_memory():
    torch.cuda.empty_cache()
    gc.collect()

# 小学1,2年生で習う漢字のリスト
grade_1_2_kanji = [
    "一", "右", "雨", "円", "王", "音", "下", "火", "花", "貝", "学", "気", "休", "金", "空", "月", "犬", "見",
    "五", "口", "校", "左", "山", "子", "四", "糸", "字", "耳", "車", "手", "十", "出", "女", "小", "上", "森",
    "人", "水", "正", "生", "青", "石", "赤", "千", "川", "先", "早", "足", "村", "大", "男", "竹", "中", "虫",
    "町", "天", "田", "土", "二", "日", "入", "年", "白", "八", "百", "文", "本", "名", "目", "立", "力", "林",
    "六", "学", "何", "作", "社", "家", "万", "海", "絵", "音", "場", "馬", "思", "姉", "島", "寺", "森", "鳥",
    "羽", "画", "工", "公", "広", "交", "光", "考", "行", "高", "黄", "合", "黒", "今", "近", "語", "午", "後",
    "行", "語", "米", "来", "楽", "間", "男", "里", "思", "会", "食", "首", "星", "雪", "黄", "森", "姉", "思",
]

# テキスト内の漢字を小学1,2年生で習う漢字のみに置き換える関数
def replace_kanji_with_hiragana(text):
    def is_valid_kanji(char):
        return char in grade_1_2_kanji

    new_text = ""
    for char in text:
        if char.isdigit() or char.isalpha():
            new_text += char
        elif re.match(r'\W', char):
            new_text += char
        elif not is_valid_kanji(char):
            new_text += f"({char})"
        else:
            new_text += char
    return new_text

# チャンクごとにテキストを生成する関数
def generate_text_in_chunks(tokenizer, model, prompt, streamer, max_length=800, chunk_size=256):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)
    generated_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=chunk_size, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1, do_sample=True, temperature=0.7, repetition_penalty=2.0)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    while len(generated_text) < max_length:
        input_ids = tokenizer.encode(generated_text, return_tensors="pt").to(device)
        attention_mask = torch.ones(input_ids.shape, device=device)
        new_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=chunk_size, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1, do_sample=True, temperature=0.7, repetition_penalty=2.0)
        new_text = tokenizer.decode(new_ids[0], skip_special_tokens=True)
        generated_text += new_text[len(generated_text):]

        if len(new_text) == len(generated_text):
            break  # これ以上生成されない場合はループを抜ける

    # 800文字に満たない場合は句読点でまとめて終了
    if len(generated_text) < max_length:
        generated_text += "。"

    return generated_text[:max_length]

# CSVファイルをUTF-16エンコーディングで読み込む
csv_file = "sentences.csv"
df = pd.read_csv(csv_file, encoding='utf-16')

# 質問のテンプレート
summary_questions = [
    "この物語の主なできごとは何ですか？",
    "この物語のしゅじんこうはだれですか？",
    "この物語の大切なところは何ですか？"
]

kanji_questions = [
    "次の言葉をかんじで書いてください：",
    "このぶんしょうの中からかんじを書きましょう：",
    "次の文の中のひらがなをかんじにかえてください："
]

knowledge_questions = [
    "この物語に出てくるしゅじんこうの名前を教えてください。",
    "この物語の中で一番大切なことは何ですか？",
    "この物語の中で、さくしゃがつたえたいことは何ですか？"
]

additional_questions = [
    "この物語のつづきをかんがえてください。",
    "この物語にでてくるどうぶつの名前をかいてください。",
    "この物語の中で、たすけがひつようだったひとはだれですか？",
    "この物語の中で、かんじをかきましょう。",
]

# 問題生成メイン関数
def generate_problems(tokenizer, model, streamer, num_problems=10):
    problems = []
    for i in range(num_problems):
        # ランダムにセンテンスを選択
        theme = df.sample(n=1).iloc[0]['sentence']
        prompt = f"{theme} このテーマに基づいて、800文字程度の物語を作成してください。物語は自然な流れで、詳細かつ興味深い内容にしてください。また、「友情」と「チームワークの大切さ」、そしてそれがどのようにしてスポーツや日常生活に影響を与えるかについても触れてみてください。「友達を大切にすること」、「協力的な関係を築くことの重要性」「自己犠牲の精神を持つことの意義」。これらのメッセージが明確に伝わることが大切です。"
        story = generate_text_in_chunks(tokenizer, model, prompt, streamer, max_length=800)  # 文章の長さを800文字に設定

        # 漢字をひらがなに置き換える
        story_with_hiragana = replace_kanji_with_hiragana(story)

        # ランダムに漢字問題を作成
        kanji_question_text = re.findall(r'\((.*?)\)', story_with_hiragana)
        if kanji_question_text:
            kanji_question_text = random.choice(kanji_question_text)
            kanji_question = f"次の言葉をかんじで書いてください：{kanji_question_text}"
        else:
            kanji_question = random.choice(kanji_questions)

        summary_question = random.choice(summary_questions)
        knowledge_question = random.choice(knowledge_questions)
        additional_question = random.choice(additional_questions)
        
        problems.append(f"テーマ:\n{theme}\n\n物語:\n{story_with_hiragana}\n\n"
                        f"1. {summary_question}\n"
                        f"2. {kanji_question}\n"
                        f"3. {knowledge_question}\n"
                        f"4. {additional_question}\n")
    
    return problems

# テキストファイル出力関数
def save_problems_to_txt(problems):
    today = datetime.today().strftime('%Y%m%d')
    filename = f"problems_{today}.txt"
    with open(filename, mode='w', encoding='utf-8') as file:
        for i, problem in enumerate(problems):
            file.write(f"問題 {i + 1}\n")
            file.write(problem)
            file.write("\n\n")
    print(f"問題生成とテキストファイルへの保存が完了しました。ファイル名: {filename}")

if __name__ == "__main__":
    model_name = "cyberagent/calm2-7b-chat-dpo-experimental"
    tokenizer, model, streamer = load_model(model_name)
    free_memory()
    problems = generate_problems(tokenizer, model, streamer, 10)  # 10の文章を生成
    save_problems_to_txt(problems)
    free_memory()
