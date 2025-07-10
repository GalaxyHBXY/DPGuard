# DPGuard
This is the official repository for the paper titled **50 Shades of Deceptive Patterns: A Unified Taxonomy, Multimodal Detection, and Security Implications** which was published at the International World Web Conference 2025 (known as WWW, A* Conference ranked by CORE)

In the repository, we have open-sourced our deceptive pattern(DP) detection model, DPGuard, along with two examples demonstrating how to use DPGuard to detect deceptive patterns.

If you have any questions, feel free to submit an issue or email me by zewei.shi@unimelb.edu.au

## Getting Started
Step 0: Install the git-lfs
* If you are on Macos, please run `brew install git-lfs && git lfs install`
* If you are on Ubuntu, please run `sudo apt-get install git-lfs && git lfs install`

Step 1: Conda Environment Setup

```
conda create -n DPGuard python==3.9.6 -y
conda activate DPGuard
```

Step 2: Required Package Installation
* Pytorch (2.2.2)
  * OSX: 
    ```
    pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
    ```
  * Linux and Windows
    ```
    # ROCM 5.7 (Linux only)
    pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/rocm5.7
    # CUDA 11.8
    pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
    # CUDA 12.1
    pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
    # CPU only
    pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
    ```
* Other Packages
    ```
    pip3 install google-genai
    pip3 install numpy==1.26
    ```

Step 3(a): OpenAI API Key Setup

```
export OPENAI_API_KEY="Your_OpenAI_API_Key"
```
step 3(b): Google Gemini API Key Setup

```
export GOOGLE_GENAI_API_KEY="Your_Google_GenAI_API_Key"
```

Step 4: Run Detection

```
python3 demo.py
```

Note: The cost of running the demo is tiny. In our test environment, It is $0.02.


## Cite Our Work

If you find our work is beneficial, please cite our work

```
@inproceedings{shi202550,
  title={50 Shades of Deceptive Patterns: A Unified Taxonomy, Multimodal Detection, and Security Implications},
  author={Shi, Zewei and Sun, Ruoxi and Chen, Jieshan and Sun, Jiamou and Xue, Minhui and Gao, Yansong and Liu, Feng and Yuan, Xingliang},
  booktitle={Proceedings of the ACM Web Conference 2025 (WWW'25)},
  year={2025}
}
```

