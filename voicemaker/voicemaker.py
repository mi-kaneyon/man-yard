import tkinter as tk
from tkinter import messagebox, filedialog
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import os

def generate_voice():
    text = text_entry.get("1.0", tk.END).strip()
    language = language_var.get()

    if not text:
        messagebox.showerror("エラー", "テキストを入力してください！")
        return

    if language not in ['ja', 'en', 'zh', 'fr']:
        messagebox.showerror("エラー", "正しい言語を選択してください！")
        return

    try:
        # ファイル名を保存するダイアログを表示
        file_path = filedialog.asksaveasfilename(defaultextension=".mp3", filetypes=[("MP3 files", "*.mp3")], title="音声ファイルの保存場所を選択してください")
        if not file_path:
            return  # キャンセルされた場合は終了

        # 音声ファイルを生成
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save("fuku_voice.mp3")

        # PyDubで音声を読み込み
        sound = AudioSegment.from_file("fuku_voice.mp3")

        # 音の高さを変更して猫の鳴き声に近づける（オクターブを上げる）
        octaves = float(octave_entry.get()) if octave_entry.get() else 1.2
        new_sample_rate = int(sound.frame_rate * (1.5 ** octaves))
        max_allowed_rate = 192000
        new_sample_rate = min(new_sample_rate, max_allowed_rate)

        # サンプルレートを変更して新しい音声を生成
        shifted_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
        shifted_sound = shifted_sound.set_frame_rate(sound.frame_rate)

        # 新しい鳴き声を保存
        shifted_sound.export(file_path, format="mp3")
        messagebox.showinfo("完了", f"音声ファイルが生成されました: {file_path}")

    except Exception as e:
        messagebox.showerror("エラー", f"音声生成中にエラーが発生しました: {str(e)}")

def play_voice():
    try:
        # 生成された音声ファイルを再生
        sound = AudioSegment.from_file("fuku_voice_shifted.mp3")
        play(sound)
    except FileNotFoundError:
        messagebox.showerror("エラー", "音声ファイルが見つかりません。まず生成してください！")
    except Exception as e:
        messagebox.showerror("エラー", f"音声再生中にエラーが発生しました: {str(e)}")

# GUIのセットアップ
root = tk.Tk()
root.title("Fukuの鳴き声生成器")
root.geometry("400x400")

# テキスト入力
text_label = tk.Label(root, text="テキストを入力してください:")
text_label.pack(pady=5)
text_entry = tk.Text(root, height=4, width=40)
text_entry.pack(pady=5)

# 言語選択
language_label = tk.Label(root, text="言語を選択してください:")
language_label.pack(pady=5)
language_var = tk.StringVar(value='ja')
language_menu = tk.OptionMenu(root, language_var, 'ja', 'en', 'zh', 'fr')
language_menu.pack(pady=5)

# オクターブ入力
octave_label = tk.Label(root, text="音の高さを入力してください（例: 1.2）:")
octave_label.pack(pady=5)
octave_entry = tk.Entry(root)
octave_entry.insert(0, "1.2")
octave_entry.pack(pady=5)

# ボタンの追加
generate_button = tk.Button(root, text="音声を生成", command=generate_voice)
generate_button.pack(pady=10)

play_button = tk.Button(root, text="音声を再生", command=play_voice)
play_button.pack(pady=5)

# メインループ
root.mainloop()
