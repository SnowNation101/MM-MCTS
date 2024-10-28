import torch
import json
from tqdm import tqdm
import re
from collections import Counter
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


path = '/fs/archive/share/u2024001021/huggingface_models/InternVL2-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

with open("datasets/we_math/testmini.json", "rb") as f:
    data = json.load(f)

output = []
width = 16

for datum in tqdm(data, desc="Inferencing"):
    img_question = datum["question"]
    img_option = datum["option"]
    img_path = "datasets/we_math/" + datum["image_path"]

    prompt = f"<image>\n\
        Now, we require you to solve a multiple-choice math question.\n\
        Please briefly describe your thought process and provide the final answer(option).\n\
        Question: {img_question}\n\
        Option: {img_option}\n\
        Regarding the format, please answer following the template below, and be sure to include two <> symbols:\n\
        <Thought process>:<<your thought process>>\n\
        <Answer>:<<your option>>."

    pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=True, temperature=1, top_k=50, top_p=0.95)

    responses = []
    for i in range(width):
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        datum["response"] = response
        responses.append(response)
    
    datum["all_responses"] = responses

    options = []
    for response in responses:
        option = response.split('Answer')[-1].strip()
        option = re.sub(r'[>><<:.]', '', option).strip()
        option = option[0] if option and option[0] in 'ABCDEFGH' else None
        options.append(option)

    datum["options"] = options

    option_response_map = {option: response for option, response in zip(options, responses)}
    most_common_option = Counter(options).most_common(1)[0][0]

    final_response = option_response_map.get(most_common_option)

    datum["response"] = final_response
    output.append(datum)

with open("output/internvl2-voting-16.json", "w") as f:
    json.dump(output, f, indent=4)