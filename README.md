# Hidden Tail Attack

This is the official repository for our paper [Hidden Tail: Adversarial Image Causing Stealthy Resource Consumption in Vision-Language Models](https://arxiv.org/abs/2508.18805).

## 1. Download Pretrained Models

First, download the required VLMs from Huggingface:

```bash
huggingface-cli download --resume-download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./LLMs/Qwen2.5-VL-7B-Instruct
huggingface-cli download --resume-download XiaomiMiMo/MiMo-VL-7B-RL --local-dir ./LLMs/MiMo-VL-7B-RL
```

## 2. Environment Setup

A complete conda environment file is provided for your convenience. To create and activate the environment, run:

```bash
conda env create -f environment.yaml
conda activate qwenvl
```

This will install all required dependencies with the correct versions.

## 3. Inference with Pre-generated Adversarial Images

A demo script `demo.ipynb` is provided to directly perform inference using pre-generated adversarial images. 

Or you can also try these images (see `./adv_images`) on https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct.


## 4. Run the Attack Script

Example for Qwen2.5-VL-7B-Instruct:

```bash
CUDA_VISIBLE_DEVICES=0 python hidden_tail_attack.py \
    --model Qwen2.5-VL-7B-Instruct \
    --image_id 0 \
    --text_id 0 \
    --maxtoken 2048 \
    --special_token im_start \
    --alpha 1000 \
    --step 5000 \
    --eos_weight 10000 \
    --epsilon 64 \
    --infer_iter 500 \
    --print_answer True
```

Example for MiMo-VL-7B-RL:

```bash
CUDA_VISIBLE_DEVICES=0 python hidden_tail_attack.py \
    --model MiMo-VL-7B-RL \
    --image_id 0 \
    --text_id 0 \
    --maxtoken 2048 \
    --special_token im_start \
    --alpha 1000 \
    --step 5000 \
    --eos_weight 10000 \
    --epsilon 64 \
    --infer_iter 500 \
    --print_answer True
```

