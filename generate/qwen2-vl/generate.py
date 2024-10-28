import json
import re
import torch
from collections import Counter
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

def load_model(model_path):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto"
    ).cuda()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def generate_responses(model, processor, data_path, width=16):
    with open(data_path+"testmini.json", "rb") as f:
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

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": data_path + datum["image_path"]},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        if width != 1:
            responses = []
            for _ in range(width):
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=1,
                    top_k=50,
                    top_p=0.95
                )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
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
            most_common_option = Counter(options).most_common(1)[0][0]
            datum["response"] = {option: response for option, response in zip(options, responses)}.get(most_common_option)
        else:
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            datum["response"] = response[0]
                
        output.append(datum)

    return output


def main():
    width = 16
    model_path = "/home/u2024001042/huggingface/Qwen/Qwen2-VL-7B-Instruct"
    data_path = "datasets/we_math/"
    output_path = f"output/qwen2vl-voting-{width}.json"

    model, processor = load_model(model_path)

    output = generate_responses(model, processor, data_path, width)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()