# Hidden Tail Attack

This is the official repository for our paper [Hidden Tail: Adversarial Image Causing Stealthy Resource Consumption in Vision-Language Models](https://arxiv.org/abs/2508.18805).

## 1. Download Pretrained Models

First, download the required VLMs from Huggingface:

```bash
huggingface-cli download --resume-download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./LLMs/Qwen2.5-VL-7B-Instruct
huggingface-cli download --resume-download XiaomiMiMo/MiMo-VL-7B-RL --local-dir ./LLMs/MiMo-VL-7B-RL
huggingface-cli download --resume-download google/gemma-3-4b-it --local-dir ./LLMs/gemma-3-4b-it
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

Or you can also try these images (see `./adv_image`) on https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct.


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
    --infer_iter 1000 \
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
    --infer_iter 1000 \
    --print_answer True
```

Example for gemma-3-4b-it:

```bash
CUDA_VISIBLE_DEVICES=0 python hidden_tail_attack.py \
    --model gemma-3-4b-it \
    --image_id 0 \
    --text_id 0 \
    --maxtoken 2048 \
    --special_token bos \
    --alpha 1000 \
    --step 5000 \
    --eos_weight 1000 \
    --epsilon 64 \
    --infer_iter 1000 \
    --print_answer True
```

## 5. Citation

If you use our code in your research, please cite:

```bibtex
@article{zhang2025hidden,
  title={Hidden Tail: Adversarial Image Causing Stealthy Resource Consumption in Vision-Language Models},
  author={Zhang, Rui and Wang, Zihan and Yang, Tianli and Li, Hongwei and Jiang, Wenbo and Zhao, Qingchuan and Liu, Yang and Xu, Guowen},
  journal={arXiv preprint arXiv:2508.18805},
  year={2025}
}
```
