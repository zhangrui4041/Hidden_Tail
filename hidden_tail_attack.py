from transformers import AutoProcessor,AutoModel,AutoModelForCausalLM
import argparse
import math
import torch
import os
import sys
import random
import pandas as pd
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from torchvision.utils import save_image
import time
from collections.abc import Iterable
from functools import lru_cache, partial
from typing import Any, Optional, TypedDict, Union
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F
import argparse
from torch.nn import CrossEntropyLoss
import re
import threading
import pynvml
import gc

import torch._dynamo
import PIL
from typing import List, Tuple, Dict
from transformers.image_processing_utils import BaseImageProcessor

torch._dynamo.config.suppress_errors = True

def calculate_visible_tokens(tokenizer, text):
    try:
        tokens_with_special = tokenizer.encode(text, add_special_tokens=False)
        
        text_without_special = tokenizer.decode(tokens_with_special, skip_special_tokens=True)
        
        visible_tokens = tokenizer.encode(text_without_special, add_special_tokens=False)
        
        return len(visible_tokens)
        
    except Exception as e:
        print(f"decode-encode failed: {e}")
        
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        special_token_ids = set()
        
        if hasattr(tokenizer, 'all_special_ids'):
            special_token_ids.update(tokenizer.all_special_ids)
        
        special_attrs = ['pad_token_id', 'eos_token_id', 'bos_token_id', 'unk_token_id', 'sep_token_id', 'cls_token_id']
        for attr in special_attrs:
            if hasattr(tokenizer, attr):
                token_id = getattr(tokenizer, attr)
                if token_id is not None:
                    special_token_ids.add(token_id)
        
        if 'SPECIAL_TOKENLIST' in globals():
            for special_token_value in SPECIAL_TOKENLIST.values():
                try:
                    token_id = tokenizer.convert_tokens_to_ids(special_token_value)
                    if token_id is not None and token_id != tokenizer.unk_token_id:
                        special_token_ids.add(token_id)
                except:
                    pass
        
        special_token_ids.discard(None)
        
        visible_tokens = [token_id for token_id in tokens if token_id not in special_token_ids]
        
        return len(visible_tokens)

class GPUMemoryMonitor:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.monitoring = False
        self.memory_records = []
        self.max_memory = 0
        self.monitor_thread = None
        
        # 初始化NVML
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            print(f"GPU {device_id} initialize success (using NVML)")
        except Exception as e:
            raise RuntimeError(f"{e}")
    
    def _monitor_memory(self):
        while self.monitoring:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                used_memory = info.used / (1024**3)  # convert to GB
                self.memory_records.append(used_memory)
                self.max_memory = max(self.max_memory, used_memory)
            except Exception as e:
                print(f"memory monitoring failed: {e}")
                break
            time.sleep(0.1)
    
    def start_monitoring(self):
        self.monitoring = True
        self.memory_records = []
        self.max_memory = 0
        self.monitor_thread = threading.Thread(target=self._monitor_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"start monitoring GPU {self.device_id} (using NVML)")
    
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if len(self.memory_records) > 0:
            avg_memory = sum(self.memory_records) / len(self.memory_records)
            min_memory = min(self.memory_records)
            return {
                'max_memory_gb': self.max_memory,
                'avg_memory_gb': avg_memory,
                'min_memory_gb': min_memory,
                'sample_count': len(self.memory_records),
                'monitoring_type': 'NVML'
            }
        else:
            return {
                'max_memory_gb': 0,
                'avg_memory_gb': 0,
                'min_memory_gb': 0,
                'sample_count': 0,
                'monitoring_type': 'NVML'
            }


def unflatten_patches(flattened_patches, grid_dims, config):
    patch_size = config.patch_size
    temporal_patch_size = config.temporal_patch_size
    merge_size = config.merge_size
    channel = 3

    grid_t, grid_h, grid_w = grid_dims

    shape_after_transpose = (
        grid_t,
        grid_h // merge_size,
        grid_w // merge_size,
        merge_size,
        merge_size,
        channel,
        temporal_patch_size,
        patch_size,
        patch_size,
    )
    patches_transposed = flattened_patches.reshape(shape_after_transpose)

    inverse_transpose_order = (0, 6, 5, 1, 3, 7, 2, 4, 8)
    patches_before_transpose = patches_transposed.transpose(inverse_transpose_order)
    
    restored_height = grid_h * patch_size
    restored_width = grid_w * patch_size
    num_frames = grid_t * temporal_patch_size
    
    image_tensor_4d = patches_before_transpose.reshape(
        num_frames, channel, restored_height, restored_width
    )
    
    return image_tensor_4d

def unnormalize_and_show(image_tensor, processor):
    img_to_show = image_tensor[0]
    
    mean = np.array(processor.image_mean).reshape(3, 1, 1)
    std = np.array(processor.image_std).reshape(3, 1, 1)
    
    unnormalized_img = img_to_show * std + mean
    unrescaled_img = unnormalized_img * 255
    rounded_img = np.round(unrescaled_img)
    clipped_img = np.clip(rounded_img, 0, 255).astype(np.uint8)
    displayable_img = clipped_img.transpose(1, 2, 0)
    
    return displayable_img


def tensor_to_pil_gemma3(
    image_tensor: torch.Tensor,
    processor: BaseImageProcessor
) -> Image.Image:
    """
    """
    if not hasattr(processor, 'image_mean') or not hasattr(processor, 'image_std'):
        raise ValueError("The provided processor is not a valid image processor.")

    reverted_tensor = image_tensor.cpu().clone()
    if reverted_tensor.dim() == 4:
        reverted_tensor = reverted_tensor.squeeze(0)

    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)
    reverted_tensor = reverted_tensor * std + mean
    reverted_tensor = reverted_tensor / processor.rescale_factor
    reverted_tensor = torch.clamp(reverted_tensor, 0, 255)

    reverted_tensor = torch.round(reverted_tensor).to(torch.uint8)
    
    reverted_tensor = reverted_tensor.permute(1, 2, 0)
    numpy_array = reverted_tensor.numpy()
    pil_image = Image.fromarray(numpy_array)
    return pil_image


parser = argparse.ArgumentParser(description="Hidden Tail Attack")
parser.add_argument("--model", type=str, default='Qwen2.5-VL-7B-Instruct')
parser.add_argument("--image_id", type=str, default="0")
parser.add_argument("--text_id", type=str, default="0")
parser.add_argument("--special_token", type=str, default="im_start")
parser.add_argument("--maxtoken", type=int, default=2048)
parser.add_argument("--alpha", type=float, default=2.0)
parser.add_argument("--eos_weight", type=float, default=0.1)
parser.add_argument("--step", type=int, default=5000)
parser.add_argument("--epsilon", type=int, default=64)
parser.add_argument("--resume_training", type=int, default=0)
parser.add_argument("--cuda_device", type=int, default=0)
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--infer_iter", type=int, default=1000)
parser.add_argument("--print_answer", type=bool, default=False)

args = parser.parse_args()
# Initialize GPU memory monitor
gpu_monitor = GPUMemoryMonitor(args.cuda_device)

temporal_patch_size = 2
patch_size = 14 
channel = 3  
merge_size = 2  

# Configure special tokens based on model
if args.model == "gemma-3-4b-it":
    SPECIAL_TOKENLIST = {
        'eos':'<end_of_turn>',
        # 'bos':'<start_of_turn>',
        'bos':'<bos>',
        'pad':'<pad>',
        'start_of_image':'<start_of_image>',
        "image_soft_token":'<image_soft_token>',
        'end_of_image':'<end_of_image>',
    }
elif args.model == "Qwen2.5-VL-7B-Instruct" or args.model == "MiMo-VL-7B-RL":
    SPECIAL_TOKENLIST = {
        "im_start":'<|im_start|>',
        "eos":'<|im_end|>',
        "object_ref_start":'<|object_ref_start|>',
        "object_ref_end":'<|object_ref_end|>',
        "box_start":'<|box_start|>',
        "box_end":'<|box_end|>',
        "quad_start":'<|quad_start|>',
        "quad_end":'<|quad_end|>',
        "vision_start":'<|vision_start|>',
        "vision_end":'<|vision_end|>',
        "vision_pad":'<|vision_pad|>',
        "image_pad":'<|image_pad|>',
        "video_pad":'<|video_pad|>',
    }

SPECIAL_TOKEN = SPECIAL_TOKENLIST[args.special_token]
EOS_TOKEN = SPECIAL_TOKENLIST['eos']
print(f"EOS_TOKEN: {EOS_TOKEN}")
print(f"SPECIAL_TOKEN: {SPECIAL_TOKEN}")
max_tokens = args.maxtoken
model_path = f"./LLMs/{args.model}"
saved_path = f"./log/{args.model}_{args.image_id}_{args.maxtoken}_{args.text_id}_{args.special_token}_{args.alpha}_{args.step}_{args.eos_weight}_{args.epsilon}"

text_path = f"./text_datasets/{args.model}_{args.image_id}_2048/output.csv" 
image_path = f"./images/{args.image_id}.jpg"

random.seed(int(time.time()))  # Initialize random seed
# ---------- Logger ----------
class Logger(object):
    def __init__(self, filename='default.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

file_name = 'process'
os.makedirs(saved_path, exist_ok=True)
sys.stdout = Logger(f'{saved_path}/{file_name}.log')

# Load model based on type
if args.model == "Qwen2.5-VL-7B-Instruct" or args.model == "MiMo-VL-7B-RL":
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLImageProcessor
    from qwen_vl_utils import process_vision_info
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="sequential"
    )
elif args.model == "gemma-3-4b-it":
    from transformers import Gemma3ForConditionalGeneration, Gemma3Processor
    
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )

print(f'Model loaded: {args.model}')
print('Using flash attention!')

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

special_token_id = processor.tokenizer(SPECIAL_TOKEN, add_special_tokens=False)['input_ids'][0]
# ---------- Load data ----------
data = pd.read_csv(text_path, header=0)
data.columns = data.columns.str.strip()
print(f"len(data): {len(data)}")
questions = data["question"].tolist()
special_token_length = 1024
answers = [str(a) for a in data["answer"]] # concatenated
# ---------- Load image and process ----------
image = Image.open(image_path).convert("RGB")  # ensure RGB format
width_pull, height_pull = image.size

# Preprocess image based on model
image_tensor = processor.image_processor(image, return_tensors="pt").pixel_values.to("cuda")

# ---------- Adversarial parameters ----------
epsilon = args.epsilon/255 # adversarial perturbation constraint
alpha = 1/255
num_iter = args.step
batch_size = 1

# resume training: find saved pt file
if args.resume_training == 1:
    pt_files = [f for f in os.listdir(saved_path) if re.match(r'adv_img_(\d+)\.pt', f)]
    if pt_files:
        steps = [int(re.findall(r'adv_img_(\d+)\.pt', f)[0]) for f in pt_files]
        max_step = max(steps)
        adv_noise_path = os.path.join(saved_path, f'adv_img_{max_step}.pt')
        adv_noise_loaded = torch.load(adv_noise_path).to("cuda")
        start_iter = max_step + 1
        print(f"detect checkpoint, load step={max_step} noise, continue training.")
    else:
        adv_noise_loaded = None
        start_iter = 0
        print("resume training but no checkpoint detected, start from scratch.")
else:
    adv_noise_loaded = None
    start_iter = 0
    print("no resume training, start from scratch.")

# initialize noise
if adv_noise_loaded is not None:
    adv_noise = adv_noise_loaded.clone().detach()
else:
    adv_noise = torch.rand_like(image_tensor).to("cuda") * 2 * epsilon - epsilon

adv_noise = adv_noise.to("cuda")
adv_noise.requires_grad_(True)
adv_noise.retain_grad()
x = image_tensor.clone().to("cuda")  

optimizer = torch.optim.Adam([adv_noise], lr=0.02)
prev_losses = [None, None, None]  # [lossANS, lossSPE, eos_penalty]
T = 2.0  # temperature coefficient
epsilon_no_zero = 1e-8  # prevent division by zero

# initialize time statistics variables
total_time = 0.0
time_records = []

for t in tqdm(range(start_iter, num_iter)):
    # record single iteration start time
    iter_start_time = time.time()
    
    target_loss = 0
    total_weighted_loss = 0  # track weighted cross entropy loss
    total_eos_penalty = 0    # track EOS penalty
    total_lossANS = 0        # track target_answer_tokens loss
    total_lossSPE = 0        # track special_answer_tokens loss

    # randomly sample a target question and answer
    for i in range(batch_size):  
        idx = random.randint(0, 39) # first 40 samples as training set
        target_answer = answers[idx]
        selected_question = questions[idx]
        special_answer = SPECIAL_TOKEN*special_token_length

        # Construct messages based on model type
        if args.model == "Qwen2.5-VL-7B-Instruct" or args.model == "MiMo-VL-7B-RL":
            messages = [
                {
                    "role": "user",
                    "content": 
                    [   
                        {"type": "text", "text": selected_question},
                        {"type": "image", "image": image_path},
                    ],
                }
            ]
            
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages) # get resized image
            
            # construct input
            inputs = processor(
                text = prompt,
                images = image_inputs,
                padding = True,
                return_tensors = "pt",
            ).to("cuda", torch.float16)
            
            inputs["pixel_values"] += adv_noise # optimized and generated use the same code
            
        elif args.model == "gemma-3-4b-it":
            messages = [
                {
                    "role": "user",
                    "content": 
                    [   
                        {"type": "text", "text": selected_question},
                        {"type": "image", "image": image_path},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to("cuda", torch.float16)
            inputs["pixel_values"] += adv_noise

        prompt_length = inputs["input_ids"].shape[1]

        # construct target answer tokens
        target_answer_tokens = processor.tokenizer(target_answer, return_tensors="pt",padding=False).input_ids.to("cuda")
        special_answer_tokens = processor.tokenizer(special_answer, return_tensors="pt",padding=False).input_ids.to("cuda")
        answer_length = target_answer_tokens.size(1)
        special_answer_length = special_answer_tokens.size(1)
        
        # concatenate input and answer
        combined_input_ids = torch.cat([inputs["input_ids"], target_answer_tokens, special_answer_tokens], dim=1)
        combined_attention_mask = torch.cat([inputs["attention_mask"], torch.ones_like(target_answer_tokens), torch.ones_like(special_answer_tokens)], dim=1) 

        labels_full = combined_input_ids.clone()
        labels_full[:, :prompt_length] = -100
        labels_ref = labels_full.clone()
        labels_full[:, :-1] = labels_ref[:, 1:] 

        # update inputs
        inputs["input_ids"] = combined_input_ids
        inputs["attention_mask"] = combined_attention_mask

        outputs = model(**inputs)
        logits = outputs.logits  # [B, L, V]

        # align labels, ignore -100
        # only calculate loss for non -100 tokens
        active_mask = (labels_full != -100)
        
        active_logits = logits[active_mask]
        active_labels = labels_full[active_mask]

        # calculate loss for each token
        loss_fct = CrossEntropyLoss(reduction='none')
        loss_per_token = loss_fct(active_logits, active_labels) # loss for each token

        # calculate loss for target_answer_tokens and special_answer_tokens
        target_answer_length = target_answer_tokens.size(1)
        special_answer_length = special_answer_tokens.size(1)

        # split loss: target_answer_tokens and special_answer_tokens
        loss_target = loss_per_token[:target_answer_length]  # loss for target_answer_tokens
        loss_special = loss_per_token[target_answer_length:target_answer_length + special_answer_length]  # loss for special_answer_tokens

        # calculate average loss for target_answer_tokens and special_answer_tokens
        lossANS = loss_target.mean()  # average loss for target_answer_tokens
        lossSPE = loss_special.mean()  # average loss for special_answer_tokens
        
        # calculate weighted total loss        
        # get EOS token id
        eos_token_id = processor.tokenizer(EOS_TOKEN, add_special_tokens=False)['input_ids'][0]
        # calculate softmax probability for all active positions
        active_probs = torch.softmax(active_logits, dim=-1)  # [num_active_tokens, vocab_size]
        
        # extract probability for EOS token
        eos_probs = active_probs[:, eos_token_id]  # [num_active_tokens]
        
        # calculate EOS token probability as penalty
        eos_penalty = eos_probs.mean()
        
        # add EOS penalty to total loss
        losses = [lossANS, lossSPE, eos_penalty]
        curr_loss_vals = [l.item() for l in losses]
        if None in prev_losses:
            weights = [1/3] * 3
        else:
            # calculate loss ratio change rate r_i
            ratios = [curr / (prev + epsilon_no_zero) for curr, prev in zip(curr_loss_vals, prev_losses)]
            
            # softmax to weight
            ratios = torch.tensor(ratios)
            weights = torch.softmax(ratios / T, dim=0).tolist()
            # add lower limit to prevent weight from being 0
            min_weight = 0.15
            weights = [max(w, min_weight) for w in weights]

            # normalize to ensure sum of weights is 1
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]
            # weights = weights.tolist()           
        prev_losses = curr_loss_vals
        loss = weights[0] * lossANS + args.alpha * weights[1] * lossSPE + args.eos_weight * weights[2] * eos_penalty

        # record losses
        total_eos_penalty += eos_penalty.item()
        total_lossANS += lossANS.item()
        total_lossSPE += lossSPE.item()
        
        target_loss += loss  # accumulate loss
    
    # calculate average loss
    target_loss /= batch_size
    total_eos_penalty /= batch_size
    total_lossANS /= batch_size
    total_lossSPE /= batch_size
    
    target_loss.requires_grad_(True)
    target_loss.backward() 

    # update noise
    with torch.no_grad():
        adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-epsilon, epsilon)  # manually update noise PGD
        adv_noise.grad.zero_()
        model.zero_grad()

    print(f"[{t}] target_loss: {target_loss.item():.4f}, lossANS: {total_lossANS:.8f}, lossSPE: {total_lossSPE:.8f}, eos_penalty: {total_eos_penalty:.8f}")
    print(f"[{t}] Loss Weights -> ANS: {weights[0]:.3f}, SPE: {weights[1]:.3f}, EOS: {weights[2]:.3f}")
    print(f"[{t}] final loss: {loss.item():.4f}, weight ANS: {weights[0] * lossANS.item():.4f}, weight SPE: {args.alpha * weights[1] * lossSPE.item():.4f}, weight EOS: {args.eos_weight * weights[2] * eos_penalty.item():.4f}")
    
    
    if (t+1) % args.infer_iter == 0: # every infer_iter iterations, do full inference
        print('######### full inference test - Iter = %d ##########' % t)        
        # save current iteration's adversarial noise
        torch.save(adv_noise, f'{saved_path}/adv_img_{t}.pt')
        print(f"adversarial noise saved: adv_img_{t}.pt")
        
        # save current iteration's image
        if args.model == "Qwen2.5-VL-7B-Instruct" or args.model == "MiMo-VL-7B-RL":
            inverse_processor = Qwen2VLImageProcessor.from_pretrained(f"./LLMs/{args.model}")

            processed_output, grid_thw = inverse_processor._preprocess(
                image,
                do_resize=inverse_processor.do_resize,
                size=inverse_processor.size,
                resample=inverse_processor.resample,
                do_rescale=inverse_processor.do_rescale,
                rescale_factor=inverse_processor.rescale_factor,
                do_normalize=inverse_processor.do_normalize,
                image_mean=inverse_processor.image_mean,
                image_std=inverse_processor.image_std,
                patch_size=inverse_processor.patch_size,
                temporal_patch_size=inverse_processor.temporal_patch_size, 
                merge_size=inverse_processor.merge_size,
                do_convert_rgb=inverse_processor.do_convert_rgb,
            )
            noise = adv_noise.to("cpu")
            noise = noise.detach().numpy()
            input = processed_output + noise
            restored_4d_tensor = unflatten_patches(
                flattened_patches=input,
                grid_dims=grid_thw,
                config=inverse_processor
            )
            displayable_img = unnormalize_and_show(restored_4d_tensor, inverse_processor)
            PIL.Image.fromarray(displayable_img).save(f"{saved_path}/adv_img_{t}.bmp")
            print(f"adversarial image saved: adv_img_{t}.bmp")
            
        elif args.model == "gemma-3-4b-it":
            inverse_processor = AutoProcessor.from_pretrained(f"./LLMs/{args.model}", trust_remote_code=True)
            inputs_for_save = processor.image_processor(images=image, return_tensors="pt")
            noise = adv_noise.to("cpu")
            noise = noise.detach().numpy()
            adv_image_tensor = inputs_for_save["pixel_values"] + noise
            restored_results = tensor_to_pil_gemma3(adv_image_tensor, processor.image_processor)
            restored_results.save(f"{saved_path}/adv_img_{t}.bmp", format="BMP")
            print(f"adversarial image saved: adv_img_{t}.bmp")
        
        print("\n========== start inference for last 20 samples ==========")
        # start memory monitoring
        gpu_monitor.start_monitoring()
        
        inference_results = []
        test_image = f"{saved_path}/adv_img_{t}.bmp"

        for idx in tqdm(range(40, min(len(questions), 60))):  # inference for last 20 samples
            inference_start_time = time.time()
            question = questions[idx]
            true_answer = answers[idx]
            
            # construct different message formats and inputs for different models
            if args.model == "Qwen2.5-VL-7B-Instruct" or args.model == "MiMo-VL-7B-RL":
                messages = [
                    {
                        "role": "user",
                        "content": 
                        [   
                            {"type": "text", "text": question},
                            {"type": "image", "image": test_image},
                        ],
                    }
                ]
                
                # process prompt
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                image_inputs, video_inputs = process_vision_info(messages) # get resized image
                
                inputs = processor(
                    text = prompt,
                    images = image_inputs,
                    padding = True,
                    return_tensors = "pt",
                ).to("cuda") 

                response = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_tokens
                )

                # decode generated text
                generated_text_true = processor.decode(
                    response[0][inputs['input_ids'].shape[-1]:], 
                    skip_special_tokens=True
                )
                generated_text_false = processor.decode(
                    response[0][inputs['input_ids'].shape[-1]:], 
                    skip_special_tokens=False
                )
                token_count = len(response[0][inputs['input_ids'].shape[-1]:])
                
            elif args.model == "gemma-3-4b-it":
                messages = [
                    {
                        "role": "user",
                        "content": 
                        [   
                            {"type": "text", "text": question},
                            {"type": "image", "image": test_image},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                ).to("cuda", dtype=torch.bfloat16)
                
                input_len = inputs["input_ids"].shape[-1]
                
                with torch.inference_mode():
                    generation = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
                    generation = generation[0][input_len:]

                    generated_text_true = processor.decode(generation, skip_special_tokens=True).strip()
                    generated_text_false = processor.decode(generation, skip_special_tokens=False).strip()
                    token_count = generation.shape[0]
                           
            inference_end_time = time.time()
            inference_duration = inference_end_time - inference_start_time            
            # save results
            result = {
                'index': idx,
                'question': question, 
                'generated_answer_true': generated_text_true,
                'generated_answer_false': generated_text_false,
                'token_count': token_count,
                "visible_tokens": calculate_visible_tokens(processor.tokenizer, generated_text_false),
                "inference_duration": inference_duration
            }
            if args.print_answer:
                print(f"[{t}] idx: {result['index']}, generated_answer_false: {result['generated_answer_false']}, \n token_count: {result['token_count']} \n visible_tokens: {result['visible_tokens']}")

            inference_results.append(result)
            
            # print progress every 10 samples
            if (idx + 1) % 10 == 0:
                print(f"inference progress: {idx + 1}/20 done")
                    
            
        # stop memory monitoring and get statistics
        memory_stats = gpu_monitor.stop_monitoring()
        

        results_df = pd.DataFrame(inference_results)
        
        # add memory statistics to DataFrame
        results_df['iter'] = t
        results_df['max_memory_gb'] = memory_stats['max_memory_gb']

        results_csv_path = f'{saved_path}/inference_results_iter_{t}.csv'
        results_df.to_csv(results_csv_path, index=False, encoding='utf-8')
        print(f"\ninference results saved to: {results_csv_path}")
        
        # save memory statistics to separate file
        memory_stats_path = f'{saved_path}/memory_stats_iter_{t}.txt'
        

        last_20_results = inference_results

        # calculate average of last 20 samples
        last_20_avg_token_count = sum(r['token_count'] for r in last_20_results) / len(last_20_results) if last_20_results else 0
        last_20_avg_visible_tokens = sum(r['visible_tokens'] for r in last_20_results) / len(last_20_results) if last_20_results else 0
        last_20_avg_inference_duration = sum(r['inference_duration'] for r in last_20_results) / len(last_20_results) if last_20_results else 0
        
        with open(memory_stats_path, 'w', encoding='utf-8') as f:
            f.write(f"=== GPU {args.cuda_device} memory usage statistics (Iter {t}) ===\n")
            f.write(f"monitoring type: {memory_stats['monitoring_type']}\n")
            f.write(f"max memory usage during inference: {memory_stats['max_memory_gb']:.2f} GB\n")
            f.write(f"average memory usage during inference: {memory_stats['avg_memory_gb']:.2f} GB\n")
            f.write(f"min memory usage during inference: {memory_stats['min_memory_gb']:.2f} GB\n")
            f.write(f"sample count: {memory_stats['sample_count']}\n")
            f.write(f"total inference duration: {inference_duration:.2f}s\n")
            f.write(f"average inference duration per sample: {inference_duration/len(inference_results):.2f}s\n\n")
            
            
            f.write(f"last 20 samples (index 40-59) average:\n")
            f.write(f"   average token count: {last_20_avg_token_count:.2f}\n")
            f.write(f"   average visible token count: {last_20_avg_visible_tokens:.2f}\n")
            f.write(f"   average inference duration: {last_20_avg_inference_duration:.3f}s\n")
            f.write(f"   sample count: {len(last_20_results)}\n\n")
            
        print(f"memory statistics saved to: {memory_stats_path}")
        
        print("=" * 80)
        print(f"iteration {t} inference done, please check the results above")
        print("=" * 80)

print("\n========== final time statistics ==========")
print(f"total training iterations: {num_iter - start_iter}")
if 'total_time' in locals():
    print(f"total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
if 'time_records' in locals() and len(time_records) > 0:
    avg_time = sum(time_records) / len(time_records)
    print(f"average time per iteration: {avg_time:.2f}s")
    print(f"fastest single iteration time: {min(time_records):.2f}s")
    print(f"slowest single iteration time: {max(time_records):.2f}s")
print("=================================")

# collect and summarize all memory statistics for inference stages
print("\n========== memory usage summary ==========")
memory_summary_files = []
for i in range(start_iter + args.infer_iter - 1, num_iter, args.infer_iter):  # find all infer_iter-multiple iterations
    memory_file = f'{saved_path}/memory_stats_iter_{i}.txt'
    if os.path.exists(memory_file):
        memory_summary_files.append(memory_file)

if memory_summary_files:
    print(f"found {len(memory_summary_files)} inference stage memory statistics files")
    
    all_max_memory = []
    all_avg_memory = []
    all_min_memory = []
    
    for file_path in memory_summary_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # parse max memory usage
                if "max memory usage during inference:" in content:
                    max_mem_line = [line for line in content.split('\n') if "max memory usage during inference:" in line][0]
                    max_memory = float(max_mem_line.split(':')[1].strip().split()[0])
                    all_max_memory.append(max_memory)
                
                # parse average memory usage  
                if "average memory usage during inference:" in content:
                    avg_mem_line = [line for line in content.split('\n') if "average memory usage during inference:" in line][0]
                    avg_memory = float(avg_mem_line.split(':')[1].strip().split()[0])
                    all_avg_memory.append(avg_memory)
                    
                # parse min memory usage
                if "min memory usage during inference:" in content:
                    min_mem_line = [line for line in content.split('\n') if "min memory usage during inference:" in line][0]
                    min_memory = float(min_mem_line.split(':')[1].strip().split()[0])
                    all_min_memory.append(min_memory)
        except Exception as e:
            print(f"error reading file {file_path}: {e}")
    
    if all_max_memory:
        print(f"all inference stages:")
        print(f"   absolute max memory usage: {max(all_max_memory):.2f} GB")
        print(f"   average max memory usage: {sum(all_max_memory)/len(all_max_memory):.2f} GB")
        print(f"   min max memory usage: {min(all_max_memory):.2f} GB")
        
    if all_avg_memory:
        print(f"   average memory usage: {sum(all_avg_memory)/len(all_avg_memory):.2f} GB")
        print(f"   max memory usage: {max(all_avg_memory):.2f} GB")
        print(f"   min memory usage: {min(all_avg_memory):.2f} GB")
        
    # save memory usage summary report
    summary_file = f'{saved_path}/memory_usage_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"=== GPU {args.cuda_device} memory usage summary report ===\n")
        f.write(f"monitoring inference iterations: {len(memory_summary_files)}\n")
        f.write(f"model: {args.model}\n")
        f.write(f"max token count: {args.maxtoken}\n")
        f.write(f"training steps: {args.step}\n")
        f.write(f"CUDA device: {args.cuda_device}\n\n")
        
        if all_max_memory:
            f.write("memory usage statistics:\n")
            f.write(f"   absolute max memory usage: {max(all_max_memory):.2f} GB\n")
            f.write(f"   average max memory usage: {sum(all_max_memory)/len(all_max_memory):.2f} GB\n")
            f.write(f"   min max memory usage: {min(all_max_memory):.2f} GB\n")
            f.write(f"   average memory usage: {sum(all_avg_memory)/len(all_avg_memory):.2f} GB\n")
            f.write(f"   max memory usage: {max(all_avg_memory):.2f} GB\n")
            f.write(f"   min memory usage: {min(all_avg_memory):.2f} GB\n\n")
            
        f.write("detailed data for each iteration:\n")
        for i, (max_mem, avg_mem, min_mem) in enumerate(zip(all_max_memory, all_avg_memory, all_min_memory)):
            iter_num = (start_iter + args.infer_iter - 1) + i * args.infer_iter
            f.write(f"  Iter {iter_num}: Max={max_mem:.2f}GB, Avg={avg_mem:.2f}GB, Min={min_mem:.2f}GB\n")
    
    print(f"memory usage summary report saved to: {summary_file}")
else:
    print("no inference stage memory statistics files found")
print("=" * 50)

# save final adversarial noise
final_adv_path = f'{saved_path}/final_adv_noise.pt'
torch.save(adv_noise, final_adv_path)
print(f"final adversarial noise saved to: {final_adv_path}")

print("\n========== training done ==========")
print("please check the inference results files for each iteration to choose the best iteration")
print(f"inference results files: {saved_path}/inference_results_iter_*.csv")
print("adversarial noise files: adv_noise_iter_*.pt")
print("final adversarial noise file: final_adv_noise.pt")
print(f"memory statistics files: {saved_path}/memory_stats_iter_*.txt")
print(f"memory usage summary file: {saved_path}/memory_usage_summary.txt")

