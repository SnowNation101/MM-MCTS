from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import json
from tqdm import tqdm
import re
from collections import Counter


processor = LlavaNextProcessor.from_pretrained(
    "/fs/archive/share/u2024001021/huggingface_models/llama3-llava-next-8b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "/fs/archive/share/u2024001021/huggingface_models/llama3-llava-next-8b-hf", 
    torch_dtype=torch.float16, device_map="auto").cuda()


with open("datasets/we_math/testmini.json", "rb") as f:
    data = json.load(f)

output = []
width = 16

for datum in tqdm(data, desc="Inferencing"):
    img_question = datum["question"]
    img_option = datum["option"]
    img_path = datum["image_path"]

    prompt = f"Now, we require you to solve a multiple-choice math question.\n\
            Please briefly describe your thought process and provide the final answer(option).\n\
            Question: {img_question}\n\
            Option: {img_option}\n\
            Regarding the format, please answer following the template below, and be sure to include two <> symbols:\n\
            <Thought process>:<<your thought process>>\n\
            <Answer>:<<your option>>."

    responses = []
    for i in range(width):
        message = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
                ],
            },
        ]
        text = processor.apply_chat_template(message, add_generation_prompt=True)

        image = Image.open("datasets/we_math/" + img_path)

        inputs = processor(
            images=image, 
            text=text, 
            return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=1024,
            do_sample=True,
            temperature=1, 
            top_k=50, 
            top_p=0.9)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids)+1:] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        responses.append(response[0])
    
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

with open("output/llava-next-voting-16.json", "w") as f:
    json.dump(output, f, indent=4)


