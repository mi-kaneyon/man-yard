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

