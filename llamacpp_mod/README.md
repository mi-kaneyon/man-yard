# llamacpp_mod
- It is utilized llama.cpp / llama-cpp-python  examples

# Install for the llama-cpp-python

> [!NOTE]
> In this example, Anaconda virtual environment.
> ~/anaconda3/envs/ldm/lib/python3.x/site-packages/llama_cpp

> [!TIP]
> Please make sure removed llama.cpp setup directory and files.

```
export CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_PATH=/usr/local/cuda-12.3 -DCUDAToolkit_ROOT=/usr/local/cuda-12.3 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12.3/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.3/lib64"
export FORCE_CMAKE=1

```
```
pip install llama-cpp-python --force-reinstall --no-cache-dir

```

> [!IMPORTANT]
> If nvidia graphic card, Driver, CUDA and cudnn are required with proper version.


# sample scripts

## CUI chat (Recommended)
- Quick response

```
python short.py

```
### example

```

[User 0001
I'm working on a project that involves recording video from my webcam using OpenCV in Python. I'm running Ubuntu and would like to create a script that captures the video stream, saves it to an AVI file, and allows me to stop the recording with a key press.
Here's what I have so far:
```python
import cv2
# Initialize video capture
cap = cv2.VideoCapture(0)  
# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Write the flipped frame
    out.write(frame)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
```
**Explanation:**
1. **Import OpenCV:** `import cv2` brings in the necessary library.
2. **Initialize Video Capture:** `cap = cv2.VideoCapture(0)` creates a capture object, assuming your webcam is at index 0. Adjust if needed.
3. **Error Handling:** It checks if the camera opened successfully.
4. **VideoWriter Setup:**
   - `fourcc = cv2.VideoWriter_fourcc(*'XVID')` selects the XVID codec for encoding (common choice).
   - `out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))` creates a VideoWriter object to save the output as 'output.avi'.
     - `20.0` is the frame rate (frames per second).
     - `(640, 480)` is the resolution.
5. **Main Loop:**
   - `ret, frame = cap.read()` captures a frame from the webcam. `ret` indicates success (True) or failure (False).
   - If `ret` is False, it assumes the stream has ended and breaks the loop.
   - `out.write(frame)` writes the captured frame to the output video file.
6. **Displaying Frames:** `cv2.imshow('frame', frame)` displays the frame in a window named 'frame'.
7. **Quit Condition:** `if cv2.waitKey(1) & 0xFF == ord('q'): break` checks if the 'q' key is pressed. If so, it breaks the loop.
8. **Cleanup:**
   - `cap.release()` releases the camera capture object.
   - `out.release()` closes the video writer.
   - `cv2.destroyAllWindows()` closes all OpenCV windows.
```
```




## GUI chat

```
python small_gemma.py

```
- using big memory size
- spend generating sentence for a long time

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.29.06              Driver Version: 545.29.06    CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3090        Off | 00000000:10:00.0  On |                  N/A |
| 53%   66C    P2             167W / 350W |  22792MiB / 24576MiB |     83%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      1988      G   /usr/lib/xorg/Xorg                          325MiB |
|    0   N/A  N/A      2169      G   /usr/bin/gnome-shell                         67MiB |
|    0   N/A  N/A     10544      G   ...irefox/4539/usr/lib/firefox/firefox      134MiB |
|    0   N/A  N/A     31358      C   python                                    22206MiB |
+---------------------------------------------------------------------------------------+


```

