"""Process the data from the output of the model to the format required 
by the evaluation script.
"""

import json


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


if __name__ == "__main__":
    process_data("output/qwen2vl-voting.json", "processed_data/qwen2vl.json")
    process_data("output/internvl2-voting.json", "processed_data/internvl2.json")
    process_data("output/llava-next-voting.json", "processed_data/llava-next.json")
