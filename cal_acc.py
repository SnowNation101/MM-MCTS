import json

cnt_correct = 0
cnt_all = 0

with open("output/scores/qwen2vl_orm.json", "r") as f:
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

print(f"Acc: {cnt_correct}/{cnt_all} = {cnt_correct/cnt_all:.2f}")