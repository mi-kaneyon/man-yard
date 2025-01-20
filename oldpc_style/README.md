# letro.py: Retro Image Effect with Dynamic Palette

This Python script applies a retro image effect to either a static image or a live webcam feed. It uses K-Means clustering to generate a dynamic 16-color palette from the input, then reduces the image to this palette and applies a pixelation effect for a retro, 8-bit look. The script leverages PyTorch for efficient color quantization, especially beneficial when using a GPU.

## Key Features

*   **Dynamic 16-Color Palette:** Generates a custom 16-color palette based on the dominant colors in the input image or webcam feed using K-Means clustering. This allows for more accurate and visually appealing color reduction compared to a fixed palette.
*   **Pixelation Effect:** Applies a pixelation effect by downscaling and then upscaling the image using nearest-neighbor interpolation, creating a blocky, retro aesthetic.
*   **GPU Acceleration:** Uses PyTorch and CUDA (if available) for accelerated color quantization, resulting in faster processing, especially for larger images or live video.
*   **Static Image and Webcam Support:** Can process either a static image provided via a command-line argument or a live webcam feed.
*   **Multiple Camera Backend Support:** Attempts to open the webcam using multiple backends (V4L2, DSHOW, GStreamer, ANY) to improve compatibility across different systems.

## Command-Line Arguments

The script accepts the following command-line argument:

*   `--image <image_path>`: Specifies the path to the input image. If this argument is provided, the script will process the image. If it's omitted, the script will use the webcam.

## Usage

**To process a static image:**

```bash
python letro.py

```
# image sample
![Test Image 3](letrosample.png)
