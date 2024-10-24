import json

def calculate_percentage(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    total_count = len(data)
    answer_in_options_count = sum(1 for item in data if item.get("answer") in item.get("options", []))
    
    percentage = (answer_in_options_count / total_count) * 100 if total_count > 0 else 0
    print(f"{file_path.split('/')[-1].split('-')[0]} Percentage of 'answer' in 'options': {percentage:.2f}%")

if __name__ == "__main__":
    calculate_percentage('output/internvl2-voting.json')
    calculate_percentage('output/qwen2vl-voting.json')