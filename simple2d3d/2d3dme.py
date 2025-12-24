import os
import sys
import time
import tempfile
import torch
import numpy as np
import gradio as gr
from PIL import Image

# --- Optimization for VRAM Fragmentation ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Path Setup ---
# Add local TripoSR directory to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
triposr_dir = os.path.join(current_dir, "TripoSR")
if triposr_dir not in sys.path:
    sys.path.append(triposr_dir)

try:
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground
    import rembg
except ImportError as e:
    print(f"Library Import Error: {e}")
    print("Please ensure 'TripoSR' folder is in the same directory and libraries are installed.")
    sys.exit(1)

# --- Configuration ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device used: {device}")

# --- Background Removal (GPU Session) ---
# Create a GPU session for rembg to speed up processing
try:
    print("Initializing GPU session for background removal (rembg)...")
    rembg_session = rembg.new_session(providers=['CUDAExecutionProvider'])
    print("GPU session created successfully.")
except Exception as e:
    print(f"Failed to create GPU session for rembg: {e}")
    print("Falling back to CPU (Process will be slower).")
    rembg_session = None

# --- Load AI Model ---
try:
    print("Loading 3D AI Model...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.to(device)
    
    # VRAM Optimization: Adjust chunk size for RTX 4070 (12GB)
    if hasattr(model, 'renderer'):
        model.renderer.chunk_size = 8192 
        print(f"VRAM Optimization: Chunk size set to {model.renderer.chunk_size}")

except Exception as e:
    print(f"Model Loading Error: {e}")
    sys.exit(1)


def fill_background(image):
    """
    Converts RGBA image to RGB with a white background.
    Necessary because the model expects 3 channels (RGB), not 4.
    """
    if image.mode == 'RGBA':
        bg = Image.new('RGB', image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    else:
        return image.convert('RGB')


def preprocess_image(input_image):
    """
    Removes background, resizes, and converts to RGB.
    """
    # Use GPU session if available
    if rembg_session:
        output = rembg.remove(input_image, session=rembg_session)
    else:
        output = rembg.remove(input_image)
        
    output = resize_foreground(output, ratio=0.85)
    final_output = fill_background(output)
    return final_output


def run_extraction(scene_codes, resolution):
    """Helper function to run mesh extraction."""
    return model.extract_mesh(scene_codes, has_vertex_color=True, resolution=resolution)


def generate_3d(input_image, do_remove_bg, progress=gr.Progress(track_tqdm=True)):
    if input_image is None:
        return None

    # Clear GPU memory before starting
    torch.cuda.empty_cache()

    try:
        # 1. Preprocessing
        progress(0.1, desc="Preprocessing & Removing Background...")
        if do_remove_bg:
            processed_image = preprocess_image(input_image)
        else:
            processed_image = resize_foreground(input_image, ratio=0.85)
            processed_image = fill_background(processed_image)

        # 2. Inference
        progress(0.4, desc="Inferring 3D Shape (AI)...")
        with torch.no_grad():
            scene_codes = model(processed_image, device=device)

        # 3. Mesh Generation (with VRAM fallback)
        progress(0.7, desc="Generating Mesh...")
        try:
            meshes = run_extraction(scene_codes, resolution=256)
        except torch.cuda.OutOfMemoryError:
            progress(0.7, desc="⚠️ VRAM Limit: Retrying with lower resolution (128)...")
            torch.cuda.empty_cache()
            meshes = run_extraction(scene_codes, resolution=128)

        # 4. Save
        progress(0.9, desc="Saving to file...")
        tmp_dir = tempfile.gettempdir()
        output_path = os.path.join(tmp_dir, f"output_{int(time.time())}.glb")
        meshes[0].export(output_path)
        
        print(f"Done: Saved to {output_path}")
        return output_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        gr.Error(f"Generation Error: {e}")
        return None
    finally:
        torch.cuda.empty_cache()

# --- GUI Setup ---
with gr.Blocks(title="2D to 3D Generator (GPU Optimized)") as demo:
    gr.Markdown("# 2D Image to 3D Model Generator")
    gr.Markdown("Upload an image to generate a 3D model (.glb). Compatible with Windows 3D Viewer and Web.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input Image", type="pil", image_mode="RGBA")
            check_bg = gr.Checkbox(label="Auto Remove Background (Recommended)", value=True)
            btn = gr.Button("Generate 3D Model", variant="primary")
        with gr.Column():
            output_3d = gr.Model3D(label="3D Result (Interactive)", clear_color=[0.0, 0.0, 0.0, 0.0])

    btn.click(fn=generate_3d, inputs=[input_img, check_bg], outputs=output_3d)

if __name__ == "__main__":
    # Launch the interface
    # server_name="0.0.0.0" allows access from other computers on the LAN
    demo.launch(inbrowser=True, server_name="0.0.0.0")
