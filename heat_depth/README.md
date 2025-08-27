# Real-time MiDaS Depth Heatmap with OpenCV and PyTorch

This project implements a real-time monocular depth estimation application using the MiDaS (Multi-Dataset Automated Segmentation) model from Hugging Face Transformers, visualized as a colored heatmap overlay on a live camera feed. It features a graphical user interface (GUI) with OpenCV trackbars to adjust the depth visualization's intensity and blending, as well as the colormap used.

The application leverages PyTorch for efficient GPU-accelerated inference and OpenCV for camera interaction and real-time display.

## Features

*   **Real-time Depth Estimation:** Utilizes a pre-trained MiDaS model (specifically `Intel/dpt-hybrid-midas`) for live monocular depth estimation.
*   **GPU Acceleration:** Optimized for NVIDIA GPUs using PyTorch and CUDA for high-performance inference.
*   **Interactive GUI:** Adjust depth visualization parameters on-the-fly using OpenCV trackbars:
    *   **Depth Strength:** Control the emphasis of depth variations in the heatmap.
    *   **Colormap:** Cycle through various OpenCV colormaps for different visual effects.
    *   **Blend Alpha:** Adjust the transparency of the heatmap overlay on the original video feed.
*   **Live Heatmap Overlay:** Displays a color-coded depth heatmap directly on the real-time camera stream.

## Prerequisites

Before running the application, ensure you have the following installed:

*   **Python 3.8+**: Recommended versions are 3.8, 3.9, or 3.10.
*   **NVIDIA GPU with CUDA support**: Essential for real-time performance.
    *   NVIDIA drivers
    *   CUDA Toolkit (compatible with your PyTorch version)
    *   cuDNN
*   **`pip`**: A working `pip` installation (version 24.x or later is recommended).

## Installation

1.  **Clone the repository (or save the script):**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git # Replace with your repo details
    cd your-repo-name
    ```
    (If you're just saving the script, navigate to its directory.)

2.  **Create and activate a virtual environment (recommended):**
    Using `conda` (if you have Anaconda/Miniconda):
    ```bash
    conda create -n midas_depth python=3.9 # Or your preferred Python version
    conda activate midas_depth
    ```
    Using `venv`:
    ```bash
    python -m venv midas_depth
    source midas_depth/bin/activate # On Windows: .\midas_depth\Scripts\activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # IMPORTANT: Adjust `cu118` to your CUDA version (e.g., cu121 for CUDA 12.1)
    pip install opencv-python numpy transformers
    ```
    *   The `transformers` library will download the MiDaS model weights the first time the script is run. An internet connection is required for the first execution.

## Usage

1.  **Save the script:**
    Save the provided Python code as `depth_heatmap_app.py` (or any `.py` name you prefer) in your project directory.

2.  **Run the application:**
    Ensure your virtual environment is activated and then execute the script:
    ```bash
    python depth_heatmap_app.py
    ```

3.  **Interact with the GUI:**
    *   Two windows will appear: "Real-time Depth Heatmap" showing the camera feed with the overlay, and "Adjustments" containing three trackbars.
    *   **Depth Strength:** Slide to increase or decrease the visual intensity of depth variations.
    *   **Colormap:** Slide to change the color scheme of the heatmap (e.g., Jet, Magma, Viridis).
    *   **Blend Alpha:** Slide to adjust the transparency of the heatmap overlay. A lower value makes the original camera feed more visible.

4.  **Exit the application:**
    Press the 'q' key on your keyboard while the "Real-time Depth Heatmap" window is active.

## Troubleshooting

*   **`TF-TRT Warning: Could not find TensorRT`**: This is a harmless warning if you are not explicitly using TensorFlow with TensorRT. The script is designed for PyTorch and CUDA. It has been suppressed in the code using `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'`.
*   **`Error: Could not open video stream.`**:
    *   Ensure your camera is properly connected and not being used by other applications (e.g., Zoom, Skype).
    *   If on Linux, verify that your user has permissions to access `/dev/video0`. You might need to add your user to the `video` group: `sudo usermod -a -G video $(whoami)` (requires log out/in or system reboot).
    *   Try different camera indices in `cv2.VideoCapture(0)` (e.g., `1`, `2`) if you have multiple cameras or if `0` doesn't work.
*   **Slow performance**: Ensure your PyTorch installation correctly utilizes your NVIDIA GPU. Check `torch.cuda.is_available()` in a Python console. If not, re-install PyTorch following the exact instructions for your CUDA version from the official PyTorch website.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Author:** Manyan3
