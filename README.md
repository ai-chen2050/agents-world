# Awesome Local AI-Agents World

This project demonstrates how to build AI agents using Langchain and Ollama.

## [AI-Agents Architecture](https://lilianweng.github.io/posts/2023-06-23-agent/)

<center><img src="./docs/img/agent-overview.png" alt="Obsidian Wechat" width="800" height="350"></center>

## [Re-Act: Reasoning and Acting](https://arxiv.org/abs/2210.03629)

<center><img src="./docs/img/ReAct.png" alt="Obsidian Wechat" width="800" height="250"></center>

## Prerequisites

1. Install [Ollama](https://ollama.com/).

2. Download the base model for Ollama.
   ```sh
   ollama pull llama3.1
   # or other models
   ollama pull nezahatkorkmaz/deepseek-v3
   ```

## Setup

0. Clone the repository from github
    ```sh
    git clone https://github.com/ai-chen2050/agents-world.git
    git submodule update --init --recursive
    ```
1. Create a new conda environment:
    ```sh
    conda create -n agents python=3.12
    conda activate agents
    ```

2. Install Langchain:
    ```sh
    pip install langchain
    # or conda
    conda install langchain -c conda-forge
    ```

## Verification

To verify the setup, execute the following script:
```sh
python scripts/ollama-chain.py
```

This will run a sample agent built with Langchain and Ollama.

## Langchain

- More Langchain info, please refer to [Langchain Cookbook](./crates/langchain/cookbook/README.md)

## Ollama
- More Ollama info, please refer to [Handy Ollama](https://github.com/datawhalechina/handy-ollama)

## Deployment

You can deploy the Ollama model using various methods. Here are some options:
- [FastAPI](https://github.com/ai-chen2050/handy-ollama/blob/main/docs/C6/1.%20%E4%BD%BF%E7%94%A8%20FastAPI%20%E9%83%A8%E7%BD%B2%20Ollama%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%AF%B9%E8%AF%9D%E7%95%8C%E9%9D%A2.md)
- [WebUI](https://github.com/ai-chen2050/handy-ollama/blob/main/docs/C6/2.%20%E4%BD%BF%E7%94%A8%20WebUI%20%E9%83%A8%E7%BD%B2%20Ollama%20%E5%8F%AF%E8%A7%86%E5%8C%96%E5%AF%B9%E8%AF%9D%E7%95%8C%E9%9D%A2.md)
- [Dify](https://github.com/ai-chen2050/handy-ollama/blob/main/docs/C7/2.%20Dify%20%E6%8E%A5%E5%85%A5%20Ollama%20%E9%83%A8%E7%BD%B2%E7%9A%84%E6%9C%AC%E5%9C%B0%E6%A8%A1%E5%9E%8B.md)
- [Dify-plus](https://github.com/YFGaia/dify-plus.git)