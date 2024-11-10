# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import json
import re
from collections import Counter
from tqdm import tqdm

import sys
import warnings

warnings.filterwarnings("ignore")
pretrained = "/fs/archive/share/llava-onevision-qwen2-72b-ov-sft"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

with open("datasets/we_math/testmini.json", "r") as f:
    data = json.load(f)

output = []

for datum in tqdm(data, desc="Inferencing"):
    prompt = f"""Now, we require you to solve a multiple-choice math question.
        Please briefly describe your thought process and provide the final answer(option).
        Question: {datum['question']}
        Option: {datum['option']}
        Regarding the format, please answer following the template below, and be sure to include two <> symbols:
        <Thought process>:<<your thought process>>
        <Answer>:<<your option>>."""
    
    image = Image.open("datasets/we_math/" + datum["image_path"])
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, 
                                      tokenizer, 
                                      IMAGE_TOKEN_INDEX, 
                                      return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]

    # Generate initial response
    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=True,
        temperature=1,
        max_new_tokens=1024,
        top_k=50,
        top_p=0.95,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

    datum["init_responses"] = text_outputs[0]

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], text_outputs[0])

    # Self correction
    sc_prompt = f"""Review your previous answer and ensure that 
                all relevant aspects of the image have been considered. 
                Are there any elements or details that you missed? Based on your
                review, improve your answer."""

    conv.append_message(conv.roles[0], sc_prompt)
    conv.append_message(conv.roles[1], None)
    prompt_reflection = conv.get_prompt()

    input_ids_reflection = tokenizer_image_token(prompt_reflection, 
                                                 tokenizer, 
                                                 IMAGE_TOKEN_INDEX, 
                                                 return_tensors="pt").unsqueeze(0).to(device)

    # Generate reflective response
    cont_reflection = model.generate(
        input_ids_reflection,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=True,
        temperature=1,
        max_new_tokens=1024,
        top_k=50,
        top_p=0.95,
    )
    reflection_outputs = tokenizer.batch_decode(cont_reflection, skip_special_tokens=True)

    datum["response"] = reflection_outputs[0]
                         
    output.append(datum)

with open("output/llava-onevision-sc.json", "w") as f:
    json.dump(output, f, indent=4)
