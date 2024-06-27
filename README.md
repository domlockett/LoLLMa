# LoLLMa 
$^{\text{ Lockett's local LLM AI}}$

This guide is tailored for data enthusiasts, gamers, and hobbyists interested in managing local Large Language Models (LLMs). 

It covers integrating various models, training methods, and configurations, providing a comprehensive setup for exploring the potential of LLMs on your local machine. 

By developing these skills and technologies for local models, you can ensure long-term usability and independence from external AI services.

This guide assumes you have a basic understanding of Python and your personal hardware specifications. The utilities needed for LLMs work across operating systems and can be run directly in a terminal.

In academic fashion, I will be keeping a general guide that includes software and utilities I used to get set up. Throughout this README you will find minimal instructions with links to additional learning materials like this beginners guide to [Learning LLM's](Learning_LLMS.md).

## Utilities

Before diving in and getting our model running take the time to understand the limits of your machine. 

### Hardware considerations

To effectively run higher-end models, a powerful computer setup is typically necessary. These models demand significant hardware resources, including high VRAM, processing power, and memory. 


- **High-End GPU**: Modern GPUs with at least 16GB VRAM, such as NVIDIA RTX 3080, 3090, or 4080, are recommended. Higher VRAM allows for running larger models and improves performance.
- **CPU**: A powerful multi-core CPU like Intel Core i9 or AMD Ryzen 9 is beneficial for handling non-offloaded tasks.
- **RAM**: At least 64GB of RAM is recommended.
- **Storage**: Fast SSD storage (preferably NVMe) to quickly load models and datasets.


### Software requirements

- **Python 3.11**: Follow this [Python Installation Guide](https://realpython.com/installing-python/).
- **CUDA Toolkit**: Follow this [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/).
- **Git**: Follow this [Git Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

Ensure that these software tools are installed and properly configured on your system. They will need access to the PATH environment variables to function correctly. Additionally, there may be other dependencies each user needs to fulfill to get these environments running smoothly.


You can check the status of these required utilities w the following commands:

```sh
python --version
nvcc --version
git --version
```

If our utilities are running smoothly we can turn to the first step of setting up our LLM which is downloading a repository that will contain the tools to make a web app that will host our LLMs.


---

### Oobabooga Text Generation WebUI

Oobabooga Text Generation WebUI is an open-source project that provides a web-based interface for interacting with various language models, including those running on llama.cpp and other backends. It allows users to easily load, configure, and generate text with different models through a user-friendly web interface. This tool is particularly useful for those who prefer a GUI over command-line interfaces.

### Key Features

- **User-Friendly Interface**: Provides a web-based interface to load models, configure settings, and generate text.
- **Backend Support**: Integrates with various backends, including llama.cpp, to run models on both CPU and GPU.
- **Model Management**: Simplifies the process of downloading, setting up, and switching between different language models.
- **Customization**: Offers advanced configuration options to optimize performance based on hardware capabilities.

Once we download the repository we will have access to three key features- llama.cpp, text-generation-webui (Oobabooga), and KoboldAI- which together create a robust environment for running local LLMs. Here's how they integrate:

1. **llama.cpp**: Serves as the backend engine that performs the actual computations required for text generation. It leverages your CPU and GPU to run models efficiently.

2. **Oobabooga Text Generation WebUI**: Acts as the primary interface where you interact with the models. It simplifies model management and provides a platform for generating text. This will open as a tab in your default web browser.

3. **KoboldAI**: Provides additional features and customization options that enhance the text generation experience. It integrates within the Oobabooga interface to offer more functionality.


## Clone Oobabooga

1. **Open Command Prompt:**
   - Press `Win + R`, type `cmd`, and press `Enter`.

2. **Navigate to directory of choice:**
   - Use the `cd` command to change the directory to where your want your `text-generation-webui` folder to be located. For example:

```sh
cd C:\<Your Username>\Documents\LocalLLM
```

3. **Clone the Repository:**
   - Clone the `text-generation-webui` repository from GitHub. Use the following command:
```sh
git clone https://github.com/oobabooga/text-generation-webui
```

4. **Navigate to the `text-generation-webui` directory:**
   - Change the directory to the newly cloned `text-generation-webui`:

```sh
cd text-generation-webui
```

5. **Set Up a Python Virtual Environment:**
   - Create and activate a Python virtual environment:
```sh
python -m venv lollma
lollma\Scripts\activate
```

4. **Install Required Dependencies:**
   - Install the necessary dependencies using `pip`:

```sh
pip install -r requirements.txt
```

## Choose a model

When selecting a model, it's crucial to consider the model's size and performance benchmarks. Here are a few leaderboards that evaluate and rank Large Language Models (LLMs):

- **Hugging Face Open LLM Leaderboard**: [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- **EleutherAI's lm-evaluation-harness**: [lm-evaluation-harness Leaderboard](https://github.com/EleutherAI/lm-evaluation-harness)
- **Papers with Code Leaderboard**: [Papers with Code Leaderboard](https://paperswithcode.com/sota)
- **OpenAI API Models**: [OpenAI Models](https://beta.openai.com/docs/models)

### Choosing a model

The main consideration when using a model is size. Choosing the appropriate model requires understanding a couple term
**Parameters:** These are the core components of an LLM. Models are often named based on the number of parameters they contain (e.g., 33B means 33 billion parameters). More parameters typically mean better performance but also higher resource requirements.

**Tokens:** These are the smallest units of text the model processes. For example, the sentence "Hello, world!" would be broken down into smaller tokens for the model to analyze.

**Quantization:** This technique reduces the model's precision to save memory. For example, reducing from 32-bit floats (FP32) to 8-bit integers (Q8) reduces the memory footprint significantly.

### Memory Usage per Parameter

| Quantization Type | Bytes per Parameter | 7B Parameters (GB) | 13B Parameters (GB) | 33B Parameters (GB) |
|-------------------|---------------------|--------------------|---------------------|---------------------|
| FP32 (32-bit float) | 4 bytes            | 28 GB              | 52 GB               | 132 GB              |
| Q8 (8-bit)          | 1 byte             | 7 GB               | 13 GB               | 33 GB               |
| Q6 (6-bit)          | 0.75 bytes         | 5.25 GB            | 9.75 GB             | 24.75 GB            |
| Q5_K_M (5-bit optimized) | 0.625 bytes    | 4.375 GB           | 8.125 GB            | 20.625 GB           |
| Q4 (4-bit)          | 0.5 bytes          | 3.5 GB             | 6.5 GB              | 16.5 GB             |

### Example Calculation for a 33B Parameter Model with Q5_K_M Quantization

1. **Quantization Type**: Q5_K_M (5-bit optimized)
2. **Bytes per Parameter**: 0.625 bytes
3. **Total Parameters**: 33 billion (33B)

**Memory Requirement Calculation**:
$$ \text{Memory Requirement} = \text{Parameters} \times \text{Bytes per Parameter} $$
$$ \text{Memory Requirement} = 33 \text{B} \times 0.625 \, \text{bytes} $$
$$ \text{Memory Requirement} = 20.625 \, \text{GB} $$
By choosing the appropriate quantization type and downloading the required files, you can optimize the performance and quality of your language model based on your hardware capabilities.


### Running the Model
Start the server with the model:

```sh
cd .\text-generation-webui
python -m venv lollma
python server.py --listen --chat --n-gpu-layers 63 

```

Access the server at [http://localhost:7860/](http://localhost:7860/). This setup will expose the server to your network, allowing connections from other computers.

## Optimizing settings

To optimize your model settings and improve the generation speed, let's first break down and adjust your settings based on your hardware specifications (RTX 4080 with 16GB VRAM). Here's a detailed table of recommended settings and an explanation for each:

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

