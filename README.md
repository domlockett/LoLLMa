# LoLLMa 
$^{\text{ Lockett's local LLM AI}}$

## Introduction to loLLMa

This project aims to help you host and manage multiple large language models (LLMs) using a gaming hardware setup, including a 4080 16GB GPU, an Intel Core i9-13900K CPU, and 128GB of RAM.

### Purpose

This guide is designed for beginners interested in handling local LLMs. It will cover how to integrate various models, training methods, and configurations. It's especially useful for data enthusiasts, researchers, and hobbyists who want to explore the potential of LLMs and set up the necessary environment on their local machines.

By developing these skills and technologies for local models, you can ensure long-term usability and independence from external AI services.

## Quick Setup Guide

This guide assumes you have a basic understanding of Python, integrated development environments (IDEs), and your personal hardware specifications. I use Windows 11 and Visual Studio Code. You will also need utilities such as Python and Conda. If you are unfamiliar with these, you may want to look for tutorials or guides online for initial setup.

### Hardware and Software Requirements:

- **Hardware**: A robust setup is recommended, such as a 4080 16GB GPU, an Intel Core i9-13900K CPU, and 128GB of RAM.
- **Software**: You need Python 3.11 and Visual Studio Code. Additionally, the CUDA Toolkit is required for GPU support.

### Setting Up the Environment:

1. **Install Python**: Download and install Python 3.11 from the [official Python website](https://www.python.org/downloads/). Verify the installation by opening a terminal and running:
    ```sh
    python --version
    ```

2. **Install Visual Studio Code and CUDA**:
   - Download and install Visual Studio Code from the [official website](https://code.visualstudio.com/).
   - During the installation of Visual Studio, ensure you select `C++ core features`, `C++ CMake Tools for Windows`, and the appropriate Windows SDK.
   - Download and install the CUDA Toolkit from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads). After installation, verify the setup by running the following commands in the terminal:
     ```sh
     nvcc --version
     ```
     ```sh
     nvidia-smi
     ```

### Additional Resources

For a more detailed guide on setting up these tools, consider referring to open-source guides available online:

- **Python Installation Guide**: [Python Installation](https://realpython.com/installing-python/)
- **Visual Studio Code Setup**: [VS Code Setup](https://code.visualstudio.com/docs/setup/setup-overview)
- **CUDA Toolkit Installation**: [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)

This guide provides a foundational setup to get you started with local LLMs. Once you have your environment ready, you can move on to integrating models and exploring various configurations to optimize performance on your hardware.

### Setting Up llama.cpp and Text Generation Web UI:
- **Clone the Repository**:
  ```cmd
  cd .\text-generation-webui
  git clone https://github.com/oobabooga/text-generation-webui.git .
  python -m venv env
  .\env\Scripts\activate
  ```
- **Additional Steps for GPU Users**:
  ```cmd
  set CMAKE_ARGS=-DLLAMA_CUBLAS=ON
  set FORCE_CMAKE=1
  pip install llama-cpp-python --no-cache-dir --verbose
  ```

### Choose and Download a Model

To choose a suitable model, refer to several leaderboards that evaluate and rank LLMs based on various benchmarks:

- **Hugging Face Open LLM Leaderboard**: [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- **EleutherAI's lm-evaluation-harness**: [lm-evaluation-harness Leaderboard](https://github.com/EleutherAI/lm-evaluation-harness)
- **Papers with Code Leaderboard**: [Papers with Code Leaderboard](https://paperswithcode.com/sota)
- **OpenAI API Models**: [OpenAI Models](https://beta.openai.com/docs/models)

For this example, we’ll use the `Nous-Hermes-2-Yi-34B` model.

### Download the Model Files

Follow these steps to download and set up your chosen models:

1. **Create Directories**: 
   Create a directory for each model in the `text-generation-webui/models` folder.
   ```cmd
   mkdir .\text-generation-webui\models\Nous-Hermes-2-Yi-34B
   ```

### Downloading the Model

To download the Nous Hermes 2 Yi 34B model with the recommended quantization types, you need to:

1. **Visit the Model Page:**
   Go to the [Nous Hermes 2 Yi 34B GGUF model page on Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B-GGUF/tree/main).

2. **Select the Desired Quantization:**
   Depending on your hardware and the desired quality-speed trade-off, choose one of the recommended quantization types (Q4_K_S, Q4_K_M, Q5_K_M).


- **Q4_K_S:** Good speed and acceptable quality, recommended for 7B models.
- **Q4_K_M:** Better quality, slightly more demanding on hardware.
- **Q5_K_M:** Highest recommended quality, slower inference, best if hardware can handle it.

To determine the memory requirements for the Nous-Hermes 33B model with Q5_K_M quantization, we need to understand how quantization affects memory usage. Here's the breakdown of the memory requirements based on the quantization type and the number of parameters:

### Understanding Quantization and Memory Usage

#### Memory Usage per Parameter
- **FP32 (32-bit float)**: 4 bytes per parameter
- **Q8 (8-bit)**: 1 byte per parameter
- **Q6 (6-bit)**: 0.75 bytes per parameter
- **Q4 (4-bit)**: 0.5 bytes per parameter
- **Q5_K_M (5-bit with specific optimizations)**: approximately 0.625 bytes per parameter (estimated based on typical quantization efficiency)

### Memory Calculation for Nous-Hermes 33B Q5_K_M
1. **Total Parameters**: 33 billion (33B)
2. **Q5_K_M Quantization**: Approximately 0.625 bytes per parameter

#### Calculation
\[ \text{Memory Requirement} = \text{Parameters} \times \text{Bytes per Parameter} \]
\[ \text{Memory Requirement} = 33B \times 0.625 \, \text{bytes} \]
\[ \text{Memory Requirement} = 20.625 \, \text{GB} \]

By choosing the appropriate quantization type and downloading the required files, you can optimize the performance and quality of your language model based on your hardware capabilities.
   Go to the model's page on Hugging Face and manually download the model files. For example, download all files for `` from [Hugging Face](https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF/tree/main/Meta-Llama-3-70B-Instruct-Q8_0.gguf).

3. **Place Files in the Directory**: 
   Move the downloaded files into the corresponding model directory you created.
   ```cmd
   move <path_to_downloaded_files>\* C:\Users\<YourUsername>\Documents\text-generation-webui\models\<ModelName>\
   ```

4. **Automatic Detection**:
   Both `kobold.cpp` and `llama.cpp` will automatically detect all model files hosted in the `models` folder. Ensure that the model directories are correctly named and files are placed in the right locations.

### Example for Meta-Llama-3-70B-Instruct-GGUF

  1. **Create Directory**:
    ```cmd
    mkdir -p C:\Users\<YourUsername>\Documents\text-generation-webui\models\Meta-Llama-3-70B-Instruct-GGUF
    ```

  2. **Download Files**:
    Visit [Meta-Llama-3-70B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF/tree/main/Meta-Llama-3-70B-Instruct-Q8_0.gguf) and download all necessary files.

  3. **Move Files**:
    ```cmd
    move <path_to_downloaded_files>\* C:\Users\<YourUsername>\Documents\text-generation-webui\models\Meta-Llama-3-70B-Instruct-GGUF\
    ```

Repeat the process for the other models by replacing the placeholders with the actual model names and download links. This ensures all models are set up and ready for use with `kobold.cpp` and `llama.cpp`.

### Running the Model
Start the server with the model:

```cmd
cd .\text-generation-webui
.\env\Scripts\activate

python server.py --listen --chat --n-gpu-layers 63 

```

Access the server at [http://localhost:7860/](http://localhost:7860/). This setup will expose the server to your network, allowing connections from other computers.

## Optimizing settings

To optimize your model settings and improve the generation speed for the Nous-Hermes-2-Yi-34B-GGUF model, let's first break down and adjust your settings based on your hardware specifications (RTX 4080 with 16GB VRAM). Here's a detailed table of recommended settings and an explanation for each:

### Model Settings

| Setting             | Current Value | Recommended Value | Explanation |
|---------------------|---------------|-------------------|-------------|
| n-gpu-layers        | 61            | 63                | Use the max GPU layers your VRAM can handle for faster processing. |
| n_ctx               | 4096          | 4096              | Context length can remain as is; adjust if you run out of memory. |
| tensor_split        | Empty         | ''                | No change needed unless using multiple GPUs. |
| n_batch             | 512           | 512               | Optimal batch size for your GPU, lower if running out of memory. |
| threads             | 0             | 16                | Set CPU threads to 16 for better multi-threading performance. |
| threads_batch       | 0             | 4                 | Set CPU batch threads to 4 to improve CPU processing efficiency. |
| alpha_value         | 1             | 1                 | Keep default unless experimenting with RoPE scaling. |
| rope_freq_base      | 5000000       | 5000000           | Keep default; used for RoPE scaling. |
| compress_pos_emb    | 1             | 1                 | Default setting for positional embeddings compression. |
| flash_attn          | Enabled       | Enabled           | Keep enabled for better attention mechanism performance on RTX cards. |
| tensorcores         | Enabled       | Enabled           | Utilize tensor cores for faster computation on RTX cards. |
| streaming_llm       | Disabled      | Disabled          | Enable if you need to stream LLM output, otherwise keep disabled. |
| attention_sink_size | 5             | 5                 | Default setting, no change needed. |
| cpu                 | Disabled      | Disabled          | Keep disabled to prefer GPU processing. |
| row_split           | Disabled      | Disabled          | Enable only if using multiple GPUs. |
| no_offload_kqv      | Disabled      | Disabled          | Allow KQV offloading to balance memory and performance. |
| no_mul_mat_q        | Disabled      | Disabled          | Keep disabled unless you face specific performance issues. |
| no_mmap             | Disabled      | Disabled          | Keep disabled unless memory mapping causes issues. |
| mlock               | Disabled      | Disabled          | Enable if your system supports it to lock model in memory. |
| numa                | Disabled      | Disabled          | Enable if using a NUMA system for memory optimization. |

### Generation Settings

| Setting                 | Current Value | Recommended Value | Explanation |
|-------------------------|---------------|-------------------|-------------|
| max_new_tokens          | 512           | 256               | Reduce max new tokens to improve generation speed. |
| temperature             | 1             | 0.7               | Adjust to 0.7 for more coherent and less random output. |
| top_p                   | 1             | 0.9               | Lower top_p to focus on the most probable tokens. |
| top_k                   | 0             | 50                | Set top_k to 50 to limit token selection to the top 50 options. |
| typical_p               | 1             | 0.9               | Lower typical_p for more coherent outputs. |
| min_p                   | 0.05          | 0.05              | Keep as is unless fine-tuning for specific output needs. |
| repetition_penalty      | 1             | 1.2               | Increase to 1.2 to avoid repetitive outputs. |
| frequency_penalty       | 0             | 0.5               | Apply a frequency penalty to reduce token repetition. |
| presence_penalty        | 0             | 0.5               | Apply a presence penalty to encourage diverse token usage. |
| tfs                     | 1             | 1                 | Keep default for truncating prompt settings. |
| truncate length         | 4096          | 2048              | Reduce truncation length to fit better within memory constraints. |
| max tokens/second       | 0             | 20                | Increase to 20 to speed up token generation. |
| max UI updates/second   | 0             | 20                | Increase to 20 for smoother UI performance. |
| seed                    | -1            | -1                | Keep as is for random seed generation. |
| activate text streaming | Enabled       | Enabled           | Keep enabled for real-time text streaming. |

By making these adjustments, you should see an improvement in generation speed and overall performance. Here are some additional tips:

1. **Monitor GPU and CPU Usage**: Use system monitoring tools to ensure your GPU and CPU are being utilized efficiently.
2. **Optimize n-gpu-layers**: Experiment with increasing or decreasing the n-gpu-layers to find the optimal setting for your VRAM capacity.
3. **Reduce Token Limits**: Lower the max_new_tokens to speed up generation time.
4. **Adjust Sampling Settings**: Fine-tune temperature, top_p, top_k, and other sampling settings to balance quality and speed.

With these optimized settings, you should be able to improve the performance of your model on your current hardware setup.

