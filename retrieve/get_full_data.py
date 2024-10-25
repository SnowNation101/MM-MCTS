import json

with open("datasets/we_math/testmini.json", "r") as f:
    wemath_data = json.load(f)

with open("similar_images_results.json", "r") as f:
    similar_images_results = json.load(f)

import pandas as pd

# Read the Parquet file
file_path = 'datasets/math_vista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet'
data = pd.read_parquet(file_path)

for datum in wemath_data:
    image = datum["image_path"]
    for result in similar_images_results:
        if result["query_image"] == image:
            retrieved_images = result["similar_images"]
            break
    
    examples = []

    for image in retrieved_images:
        row = data[data["pid"] == image]

        question = row["question"]
        image = row["image"]
        choices = row["choices"]
        


    datum["retrieved_images"] = retrieved_images
        



    print(retrieved_images)


# # Access individual columns
# for index, row in data.iterrows():
#     question = row['question']
#     image_path = row['image']
#     choices = row['choices']
#     unit = row['unit']
#     precision = row['precision']
#     answer = row['answer']
#     question_type = row['question_type']
#     answer_type = row['answer_type']
#     pid = row['pid']
#     metadata = row['metadata']
#     query = row['query']

#     # Print or process the data as needed
#     print(f"Question ID: {pid}")
#     print(f"Question: {question}")
#     print(f"Image Path: {image_path}")
#     print(f"Choices: {choices}")
#     print(f"Answer: {answer}")
#     print(f"Metadata: {metadata}")
#     print()