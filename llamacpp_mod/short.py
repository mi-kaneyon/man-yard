
import torch
from llama_cpp import Llama

model_path = "./gemma-2-27b-it-Q4_K_M.gguf"

# GPU layer setting
n_gpu_layers = -1  # All layer is transfered to GPU

# Loading model
llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=n_gpu_layers, verbose=True, n_batch=32)  # batch size 32

# setting prompt
input_text = input_text = "what is my favorite food？"
response = llm(input_text, max_tokens=2000, stop=None, echo=False)  # max_tokens set 2000

# response format
formatted_response = response['choices'][0]['text'].replace('<eos>', '').replace('<bos>', '').strip()
formatted_response = formatted_response.replace('\n\n', '\n').replace('。', '。\n')

print(formatted_response)

# cleanup model
llm.close()



