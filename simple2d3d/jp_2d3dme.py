import os
import sys
import time
import tempfile
import torch
import numpy as np
import gradio as gr
from PIL import Image

# --- 【VRAM対策1】メモリ断片化を防ぐ設定 ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- パス設定 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
triposr_dir = os.path.join(current_dir, "TripoSR")
if triposr_dir not in sys.path:
    sys.path.append(triposr_dir)

try:
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground
    import rembg
except ImportError as e:
    print(f"ライブラリ読み込みエラー: {e}")
    sys.exit(1)

# --- Configuration ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")

# モデル読み込み
try:
    print("モデルを読み込んでいます...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.to(device)
    
    # --- 【VRAM対策2】チャンクサイズ（一度に処理する量）を小さくする ---
    # デフォルトは大きすぎる場合があるため、小さく設定して分割処理させます
    if hasattr(model, 'renderer'):
        # 8192 は安全圏。それでも落ちる場合は 4096 や 2048 に下げてください
        model.renderer.chunk_size = 4096 
        print(f"VRAM節約モード: Chunk size set to {model.renderer.chunk_size}")

except Exception as e:
    print(f"モデル読み込みエラー: {e}")
    sys.exit(1)


def fill_background(image):
    """RGBA画像を白背景のRGB画像に変換"""
    if image.mode == 'RGBA':
        bg = Image.new('RGB', image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    else:
        return image.convert('RGB')


def preprocess_image(input_image):
    if input_image is None:
        return None
    # 背景削除 -> リサイズ -> RGB変換
    output = rembg.remove(input_image)
    output = resize_foreground(output, ratio=0.85)
    final_output = fill_background(output)
    return final_output


def run_extraction(scene_codes, resolution):
    """メッシュ抽出を実行するヘルパー関数"""
    return model.extract_mesh(scene_codes, has_vertex_color=True, resolution=resolution)


def generate_3d(input_image, do_remove_bg):
    if input_image is None:
        return None

    # メモリ掃除
    torch.cuda.empty_cache()

    try:
        print("--- 生成プロセス開始 ---")
        
        # 1. 前処理
        if do_remove_bg:
            print("処理中: 背景削除と正規化...")
            processed_image = preprocess_image(input_image)
        else:
            print("処理中: 画像フォーマット変換(RGB)...")
            processed_image = resize_foreground(input_image, ratio=0.85)
            processed_image = fill_background(processed_image)

        # 2. 推論
        print("処理中: 3D形状を推論中 (AI)...")
        with torch.no_grad():
            scene_codes = model(processed_image, device=device)

        # 3. メッシュ生成（自動リトライ機能付き）
        print("処理中: メッシュ抽出中 (分割処理)...")
        try:
            # まず高解像度(256)でトライ
            meshes = run_extraction(scene_codes, resolution=256)
        except torch.cuda.OutOfMemoryError:
            print("⚠️ VRAM不足が発生しました。解像度を下げて再試行します (256 -> 128)")
            torch.cuda.empty_cache() # メモリ解放
            try:
                # 解像度を落としてリトライ
                meshes = run_extraction(scene_codes, resolution=128)
            except torch.cuda.OutOfMemoryError:
                 print("⚠️ 再度VRAM不足が発生。さらに解像度を下げます (128 -> 64)")
                 torch.cuda.empty_cache()
                 meshes = run_extraction(scene_codes, resolution=64)

        # 4. 保存
        tmp_dir = tempfile.gettempdir()
        output_path = os.path.join(tmp_dir, f"output_{int(time.time())}.glb")
        meshes[0].export(output_path)
        
        print(f"完了: {output_path}")
        return output_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"生成エラー: {e}")
        return None
    finally:
        # 処理が終わったら必ずメモリを掃除
        torch.cuda.empty_cache()

# --- GUI Setup ---
with gr.Blocks(title="2D to 3D Generator (Low VRAM Mode)") as demo:
    gr.Markdown("# 2D Image to 3D Model Generator")
    gr.Markdown("VRAM最適化版: 画像をアップロードして生成してください。")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="入力画像", type="pil", image_mode="RGBA")
            check_bg = gr.Checkbox(label="背景を自動削除する", value=True)
            btn = gr.Button("3Dモデル生成", variant="primary")
        with gr.Column():
            output_3d = gr.Model3D(label="3D結果", clear_color=[0.0, 0.0, 0.0, 0.0])

    btn.click(fn=generate_3d, inputs=[input_img, check_bg], outputs=output_3d)

if __name__ == "__main__":
    # GPUエラー回避用(rembg)
    os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"
    
    demo.launch(inbrowser=True, server_name="0.0.0.0")
