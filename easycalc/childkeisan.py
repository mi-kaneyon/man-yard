import tkinter as tk
from tkinter import font
import random
import json

# --- グローバル設定 ---
current_lang = "ja"  # "ja" または "en" を設定

# --- ファイルから問題データを読み込む ---
def load_problem_data(filename="problems.json"):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"エラー: {filename} が見つかりません。")
        return None
    except json.JSONDecodeError:
        print(f"エラー: {filename} の形式が正しくありません。")
        return None

# --- 問題を生成するクラス（機能拡張版） ---
class ProblemGenerator:
    def __init__(self, lang, level, problem_data):
        self.lang_data = problem_data[lang]
        
        min_val = 10**(level - 1)
        max_val = (10**level) - 1
        self.num_range = (min_val, max_val)

    def generate_word_problem(self):
        """文章問題を生成する (従来の機能)"""
        template_data = random.choice(self.lang_data["templates"])
        operation = template_data["operation"]
        # (以下、前回と同じロジック...)
        min_val, max_val = self.num_range
        num1 = random.randint(min_val, max_val)
        num2 = random.randint(min_val, max_val)

        if operation == "subtract":
            if num1 < num2: num1, num2 = num2, num1
        elif operation == "divide":
            if num2 == 0: num2 = 1
            num1 = num1 * num2
            if num1 > max_val :
                divisor = random.randint(min_val if min_val > 0 else 1, int(max_val**0.5))
                answer = random.randint(min_val if min_val > 0 else 1, max_val // divisor)
                num1 = divisor * answer
                num2 = divisor
        
        # 答えを計算
        if operation == "add": correct_answer = num1 + num2
        elif operation == "subtract": correct_answer = num1 - num2
        elif operation == "multiply": correct_answer = num1 * num2
        elif operation == "divide": correct_answer = num1 // num2
        else: correct_answer = 0

        question_text = template_data["format"].format(
            person=random.choice(self.lang_data["people"]),
            item=random.choice(self.lang_data["items"]),
            action=random.choice(self.lang_data.get("actions", [""])),
            num1=num1,
            num2=num2
        )
        return question_text, correct_answer

    def generate_calculation(self, require_carry=False):
        """単純な計算式を生成する（繰り上がり・繰り下がり機能付き）"""
        min_val, max_val = self.num_range
        operation = random.choice(['add', 'subtract'])
        
        # 繰り上がり・繰り下がりを保証するためのループ
        # （より高度な生成方法もあるが、確実性を優先）
        while True:
            num1 = random.randint(min_val, max_val)
            num2 = random.randint(min_val, max_val)

            if operation == 'add':
                if not require_carry or (num1 % 10 + num2 % 10 >= 10):
                    question_text = f"{num1} + {num2} = ?"
                    correct_answer = num1 + num2
                    break
            
            elif operation == 'subtract':
                if num1 < num2: num1, num2 = num2, num1 # 答えがマイナスにならないように
                if not require_carry or (num1 % 10 < num2 % 10):
                    question_text = f"{num1} - {num2} = ?"
                    correct_answer = num1 - num2
                    break
                    
        return question_text, correct_answer

# --- ゲームのGUIとメインロジック ---
class CalculationGame:
    def __init__(self, root, level, problem_data):
        self.root = root
        self.lang_texts = LANGUAGES[current_lang]
        self.problem_generator = ProblemGenerator(current_lang, level, problem_data)
        
        self.setup_ui()
        self.new_word_problem() # 最初の問題は文章問題にする

    def setup_ui(self):
        self.root.title(self.lang_texts["title"])
        self.root.geometry("600x400")

        # フォント設定
        self.custom_font = font.Font(family=self.lang_texts["font_family"], size=self.lang_texts["font_size"])
        self.result_font = font.Font(family=self.lang_texts["font_family"], size=self.lang_texts["font_size"] + 4, weight="bold")
        
        # --- UI要素の配置 ---
        # 問題表示ラベル
        self.question_label = tk.Label(self.root, text="", font=self.custom_font, pady=20, justify=tk.LEFT, wraplength=580, height=4)
        self.question_label.pack(fill=tk.X, padx=10)

        # 回答入力フレーム
        input_frame = tk.Frame(self.root)
        input_frame.pack(pady=10)
        tk.Label(input_frame, text=self.lang_texts["answer_label"], font=self.custom_font).pack(side=tk.LEFT)
        self.answer_entry = tk.Entry(input_frame, font=self.custom_font, width=10)
        self.answer_entry.pack(side=tk.LEFT)
        
        # 「こたえあわせ」ボタン
        self.submit_button = tk.Button(self.root, text=self.lang_texts["submit_button"], font=self.custom_font, command=self.check_answer)
        self.submit_button.pack(pady=5)

        # 結果表示ラベル
        self.result_label = tk.Label(self.root, text="", font=self.result_font, pady=15, height=2)
        self.result_label.pack()

        # 問題生成ボタンをまとめるフレーム
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        tk.Button(button_frame, text=self.lang_texts["word_problem_btn"], font=self.custom_font, command=self.new_word_problem).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text=self.lang_texts["calc_problem_btn"], font=self.custom_font, command=self.new_calculation_problem).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text=self.lang_texts["carry_problem_btn"], font=self.custom_font, command=lambda: self.new_calculation_problem(require_carry=True)).pack(side=tk.LEFT, padx=5)
        
        # 終了ボタン
        exit_button_frame = tk.Frame(self.root)
        exit_button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        tk.Button(exit_button_frame, text=self.lang_texts["exit_button"], command=self.root.destroy).pack(side=tk.RIGHT)

    def _update_question(self, question_text, correct_answer):
        """問題と答えを画面にセットする共通処理"""
        self.question_label.config(text=question_text)
        self.correct_answer = correct_answer
        self.answer_entry.delete(0, tk.END)
        self.result_label.config(text="")

    def new_word_problem(self):
        """文章問題を生成して表示"""
        question_text, correct_answer = self.problem_generator.generate_word_problem()
        self._update_question(question_text, correct_answer)

    def new_calculation_problem(self, require_carry=False):
        """計算問題を生成して表示"""
        question_text, correct_answer = self.problem_generator.generate_calculation(require_carry=require_carry)
        self._update_question(question_text, correct_answer)

    def check_answer(self):
        """ユーザーの回答をチェックする"""
        try:
            user_answer = int(self.answer_entry.get())
            if user_answer == self.correct_answer:
                self.result_label.config(text=self.lang_texts["correct_message"], fg="blue")
            else:
                self.result_label.config(text=f'{self.lang_texts["incorrect_message"]} (こたえは {self.correct_answer})', fg="red")
        except ValueError:
            self.result_label.config(text=self.lang_texts["enter_number_message"], fg="red")

# --- 難易度選択画面（変更なし） ---
class LevelSelector:
    def __init__(self, root, problem_data):
        self.root = root
        self.problem_data = problem_data
        self.root.title("難易度を選んでね")
        self.root.geometry("300x250")
        tk.Label(root, text="レベルを選択", font=("Meiryo UI", 16)).pack(pady=20)
        tk.Button(root, text="かんたん (1桁の計算)", font=("Meiryo UI", 12), command=lambda: self.start_game(1)).pack(pady=5, fill=tk.X, padx=20)
        tk.Button(root, text="ふつう (2桁の計算)", font=("Meiryo UI", 12), command=lambda: self.start_game(2)).pack(pady=5, fill=tk.X, padx=20)
        tk.Button(root, text="むずかしい (3桁の計算)", font=("Meiryo UI", 12), command=lambda: self.start_game(3)).pack(pady=5, fill=tk.X, padx=20)

    def start_game(self, level):
        self.root.destroy()
        game_root = tk.Tk()
        CalculationGame(game_root, level, self.problem_data)
        game_root.mainloop()

# --- プログラムの実行部分 ---
if __name__ == "__main__":
    # UIテキストにボタンの文言を追加
    LANGUAGES = {
        "ja": {
            "title": "けいさん学習ゲーム", "answer_label": "こたえ:", "submit_button": "こたえあわせ",
            "correct_message": "せいかい！", "incorrect_message": "ざんねん", "enter_number_message": "すうじをいれてね",
            "font_family": "Meiryo UI", "font_size": 14,
            "word_problem_btn": "文章問題", "calc_problem_btn": "計算問題", "carry_problem_btn": "繰り上がり計算",
            "exit_button": "おわる"
        },
        "en": {
            "title": "Calculation Learning Game", "answer_label": "Answer:", "submit_button": "Check",
            "correct_message": "Correct!", "incorrect_message": "Wrong", "enter_number_message": "Enter a number",
            "font_family": "Arial", "font_size": 11,
            "word_problem_btn": "Word Problem", "calc_problem_btn": "Calculation", "carry_problem_btn": "Carry Calc",
            "exit_button": "Exit"
        }
    }
    
    problem_data = load_problem_data()
    if problem_data:
        root = tk.Tk()
        app = LevelSelector(root, problem_data)
        root.mainloop()
