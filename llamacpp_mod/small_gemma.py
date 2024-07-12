import os
import tkinter as tk
from tkinter import scrolledtext
import threading
from llama_cpp import Llama
import torch

# GPU usability check
use_gpu = torch.cuda.is_available()
print(f"GPU Available: {use_gpu}")

# model path (local)
model_path = "/media/kanengi/0a4bbc0b-5d3f-479e-8bc4-b087704b7b5d/gemma/gemma-2-27b-it-gguf/gemma-2-27b-it-bf16-00001-of-00002.gguf"

# checking model path
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model path does not exist: {model_path}")

# GPU layer setting
n_gpu_layers = 20  # adjust as your environment

# Loading model
llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=n_gpu_layers, verbose=True, n_batch=32)  # e.g. batch size 32

def generate_response(prompt):
    try:
        response = llm(prompt, max_tokens=150, stop=None, echo=False)
        formatted_response = response['choices'][0]['text'].replace('<eos>', '').replace('<bos>', '').strip()
        formatted_response = formatted_response.replace('\n\n', '\n').replace('。', '。\n')
        return formatted_response
    except Exception as e:
        return f"Error: {str(e)}"

# GUI application setting
class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot")

        # font setting
        font_style = ("Arial", 14)

        self.chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=25, font=font_style)
        self.chat_window.grid(column=0, row=0, padx=10, pady=10)

        self.user_input = tk.Entry(root, width=70, font=font_style)
        self.user_input.grid(column=0, row=1, padx=10, pady=10)

        self.send_button = tk.Button(root, text="Send", command=self.send_message, font=font_style)
        self.send_button.grid(column=0, row=2, padx=10, pady=10)

    def send_message(self):
        user_message = self.user_input.get()
        if user_message:
            self.chat_window.insert(tk.END, "You: " + user_message + "\n")
            self.user_input.delete(0, tk.END)
            threading.Thread(target=self.generate_and_display_response, args=(user_message,)).start()

    def generate_and_display_response(self, user_message):
        bot_response = generate_response(user_message)
        self.chat_window.insert(tk.END, "Bot: " + bot_response + "\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
