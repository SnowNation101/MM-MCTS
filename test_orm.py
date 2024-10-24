import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def score_orm(model_name, data_path, output_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(data_path, "r") as f:
        data = json.load(f)

    outputs = []

    for datum in tqdm(data): 
        question = datum["question"]
        predictions = datum["prediction_paths"]
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

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            orm_scores.append(response)

        datum["orm_scores"] = orm_scores
        outputs.append(datum)

    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=4)

def score_prm(model_name, data_path, output_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(data_path, "r") as f:
        data = json.load(f)

    outputs = []

    for datum in tqdm(data): 
        question = datum["question"]
        predictions = datum["prediction_paths"]
        prm_scores = []
        for prediction in predictions:
            predict_sentences = prediction.split(".")
            n = len(predict_sentences)
            scores = []
            for i in range(n):
                prompt = f"{question}{'.'.join(predict_sentences[:i+1])}"

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
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                scores.append(int(response))
            
            prm_scores.append(sum(scores)/n)

        datum["prm_scores"] = prm_scores
        outputs.append(datum)

    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=4)


def cal_orm_acc(file_path):
    cnt_correct = 0
    cnt_all = 0

    with open(file_path, "r") as f:
        data = json.load(f)

    for datum in data:
        standard = datum["standard_answer"]
        answers = datum["extracted_answers"]
        scores = datum["orm_scores"]

        for answer, score in zip(answers, scores):
            if answer == standard and score == "1":
                cnt_correct += 1
                break
        
        cnt_all += 1

    print(f"Acc: {cnt_correct}/{cnt_all} = {cnt_correct*100/cnt_all:.2f}")


def generate_seudo_orm_data(data_path, output_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    with open("datasets/we_math/testmini.json", "r") as f:
        data0 = json.load(f)

    outputs = []
    for new, source in zip(data, data0):
        choose = ""
        for ans, orm in zip(new["extracted_answers"], new["orm_scores"]):
            if orm == "1" and ans == source["answer"]:
                choose = ans
            response = f"<Thought process>: xxxxxxxxxxxx. <Answer>: {choose}.xxxxxx"
        source["response"] = response
        outputs.append(source)

    
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=4)


def generate_seudo_prm_data(data_path, output_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    with open("datasets/we_math/testmini.json", "r") as f:
        data0 = json.load(f)

    outputs = []
    for new, source in zip(data, data0):
        prm_scores = new["prm_scores"]
        max_index = prm_scores.index(max(prm_scores))
        choose = new["extracted_answers"][max_index]
        response = f"<Thought process>: xxxxxxxxxxxx. <Answer>: {choose}.xxxxxx"
        source["response"] = response
        outputs.append(source)

    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=4)


if __name__ == "__main__":
    model_name = "/fs/archive/share/mcts_models/Qwen2.5-7B-Instruct_wemath_mcts_1022_prm"
    # model_name = "/fs/archive/share/mcts_models/Qwen2.5-7B-Instruct_mathvista_mcts_1017_orm"


    # score_orm(model_name, "processed_data/internvl2.json", "output/scores/internvl2_orm.json")
    score_prm(model_name, "processed_data/qwen2vl.json", "output/scores/qwen2vl_prm.json")
    
    # cal_orm_acc("output/scores/qwen2vl_orm.json")

    # generate_seudo_data("output/scores/qwen2vl_orm_45.json", "output/scores/qwen2vl_orm_seudo_45.json")