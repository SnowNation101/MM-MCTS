"""Process the data from the output of the model to the format required 
by the evaluation script.
"""

import json
import os
import re
from collections import Counter


def process_data(in_path, out_path):
    with open(in_path, "r") as f:
        data = json.load(f)

    output = []

    for datum in data:
        img_question = datum["question"]
        img_option = datum["option"]
        img_path = datum["image_path"]

        prompt = (
            f"Now, we require you to solve a multiple-choice math question.\n"
            f"Please briefly describe your thought process and provide the final answer(option).\n"
            f"Question: {img_question}\n"
            f"Option: {img_option}\n"
            f"Regarding the format, please answer following the template below, and be sure to include two <> symbols:\n"
            f"<Thought process>:<<your thought process>>\n"
            f"<Answer>:<<your option>>"
        )

        output.append({
            "question": prompt,
            "standard_answer": datum["answer"],
            "prediction_paths": datum["all_responses"],
            "extracted_answers": datum["options"]
        })

    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)


def process_gpt_output(in_dir, out_path):
    input = []
    output = []
    for filename in os.listdir(in_dir):
        if filename.endswith(".json"):
            with open(os.path.join(in_dir, filename), "r") as f:
                data = json.load(f)
                input.append(data)
    
    n = len(input[0])
    for i in range(n):
        datum = input[0][i]
        responses = []
        for j in range(len(input)):
            responses.append(input[j][i]["response"])
        output.append(datum)
        datum["all_responses"] = responses
        options = []
        for response in responses:
            option = response.split('Answer')[-1].strip()
            option = re.sub(r'[>><<:.]', '', option).strip()
            option = option[0] if option and option[0] in 'ABCDEFGH' else None
            options.append(option)

        datum["options"] = options
        most_common_option = Counter(options).most_common(1)[0][0]
        datum["option"] = most_common_option
        datum["response"] = {option: response for option, response in zip(options, responses)}.get(most_common_option)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)


def main():
    process_gpt_output("output/gpt4o", "output/gpt4o-voting-4.json")
    process_gpt_output("output/gpt4v", "output/gpt4v-voting-4.json")


if __name__ == "__main__":
    main()
    # process_data("output/qwen2vl-voting.json", "processed_data/qwen2vl.json")
    # process_data("output/internvl2-voting.json", "processed_data/internvl2.json")
    # process_data("output/llava-next-voting.json", "processed_data/llava-next.json")
    