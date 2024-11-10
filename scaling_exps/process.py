import json
import random
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter


def get_mv_data(model_name, scaling):
    f = open(f'{model_name}/majority_voting/{model_name}-32.json')
    data = json.load(f)

    output = []
    for datum in data:
        responses = datum['all_responses']
        responses = random.sample(responses, scaling)
        datum['all_responses'] = responses

        options = []
        for response in responses:
            option = response.split('Answer')[-1].strip()
            option = re.sub(r'[>><<:.]', '', option).strip()
            option = option[0] if option and option[0] in 'ABCDEFGH' else None
            options.append(option)

        datum["options"] = options
        most_common_option = Counter(options).most_common(1)[0][0]
        datum["response"] = {option: response for option, response in zip(options, responses)}.get(most_common_option)

        output.append(datum)

    with open(f'{model_name}/majority_voting/{model_name}-{scaling}.json', 'w') as f:
        json.dump(output, f, indent=4)

    
def run_orm(model, tokenizer, data_path, output_path):
    with open(data_path, "r") as f:
        data = json.load(f)

    outputs = []

    for datum in tqdm(data): 
        prompt = f"""Now, we require you to solve a multiple-choice math question.
        Please briefly describe your thought process and provide the final answer(option).
        Question: {datum['question']}
        Option: {datum['option']}
        Regarding the format, please answer following the template below, and be sure to include two <> symbols:
        <Thought process>:<<your thought process>>
        <Answer>:<<your option>>."""
        question = prompt
        predictions = datum["all_responses"]
        orm_scores= []
        for prediction in predictions:
            prompt = f"{question}\n{prediction}"
            
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=50
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            score = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            orm_scores.append(score)

        datum["orm_scores"] = orm_scores
        
        selected_responses = []
        for score, prediction in zip(orm_scores, predictions):
            if score.strip() == "1":
                selected_responses.append(prediction)

        response = random.choice(predictions)
        if selected_responses != []:
            response = random.choice(selected_responses)

        datum["response"] = response
        outputs.append(datum)

    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=4)
    

def get_rand_data(model_name, data_path, scaling):
    f = open(data_path)
    data = json.load(f)

    output = []
    for datum in data:
        responses = datum['all_responses']
        response = random.choice(responses)
        datum["response"] = response
        output.append(datum)
    
    with open(f'{model_name}/random/{model_name}-{scaling}.json', 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    
    model_name = "qwen2vl"
    # get_mv_data(model_name, 2)
    # get_mv_data(model_name, 4)
    # get_mv_data(model_name, 8)


    # get_mv_data(model_name, 8)

    model_path = "/fs/archive/share/mcts_models/Qwen2.5-7B-Instruct_wemath_mcts_1022_orm"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    
    for scaling in [2, 4, 8, 16, 32]:
        run_orm(model, tokenizer, 
                f"{model_name}/majority_voting/{model_name}-{scaling}.json", 
                f"{model_name}/orm/{model_name}-{scaling}.json")
        print(f"Done with {scaling}")

    # for scaling in [2, 4, 8, 16, 32]:
    #     get_rand_data(model_name, 
    #                   f"{model_name}/majority_voting/{model_name}-{scaling}.json",
    #                   scaling)