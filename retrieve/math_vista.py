"""Use MathVista as questions to retrieve from MathVerse, MathVision,
and WeMath datasets.
"""

import clip
import torch
import faiss
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image
from utils.retrieval import image_to_image, text_to_text

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_ids = list(range(torch.cuda.device_count()))

def retrieve_wemath(model, preprocess, image_index, text_index, test_dataset):
    """ Retrieve similar images and text from WeMath dataset.
    Returns the retrieved data.

    Args:
        model (CLIP): CLIP model.
        preprocess (torchvision.transforms): Image preprocessing pipeline.
        image_index (faiss.Index): Faiss index for image embeddings.
        text_index (faiss.Index): Faiss index for text embeddings.
        test_dataset (list): List of test data.
    """

    with open('datasets/we_math/testmini.json') as f:
        data = json.load(f)

    i2i_results = {}
    t2t_results = {}

    for pid, item in tqdm(test_dataset.items(), desc="Retrieving from WeMath"):        
        image_path = item['image']
        image = Image.open('datasets/math_vista/' + image_path)
        D, I = image_to_image(image, model, preprocess, image_index, topk=50)
        new_item = item.copy()
        results = []
        for index, distance in zip(I[0], D[0]):
            result = {}
            result['image_path'] = data[index]['image_path']
            result['quesiton'] = data[index]['question']
            result['answer'] = data[index]['answer']
            choices = data[index]['option']
            result['metadata'] = {
                "distance": float(distance), 
                "choices": choices,
                }
            results.append(result)

        new_item['retrieved_data'] = results
        i2i_results[pid] = new_item

        question = item['question']
        D, I = text_to_text(question, model, text_index, topk=50)
        new_item = item.copy()
        results = []
        for index, distance in zip(I[0], D[0]):
            result = {}
            result['image_path'] = data[index]['image_path']
            result['quesiton'] = data[index]['question']
            result['answer'] = data[index]['answer']
            choices = data[index]['option']
            result['metadata'] = {
                "distance": float(distance), 
                "choices": choices,
                }
            results.append(result)

        new_item['retrieved_data'] = results
        t2t_results[pid] = new_item

    return i2i_results, t2t_results


def retrieve_mathvision(model, preprocess, image_index, text_index, test_dataset):
    """ Retrieve similar images and text from MathVision dataset.
    Returns the retrieved data.

    Args:
        model (CLIP): CLIP model.
        preprocess (torchvision.transforms): Image preprocessing pipeline.
        image_index (faiss.Index): Faiss index for image embeddings.
        text_index (faiss.Index): Faiss index for text embeddings.
        test_dataset (list): List of test data.
    """
    
    data = pd.read_parquet('datasets/math_vision/data/'
                           'test-00000-of-00001-3532b8d3f1b4047a.parquet')

    i2i_results = {}
    t2t_results = {}

    for pid, item in tqdm(test_dataset.items(), desc="Retrieving from MathVision"):
        image_path = item['image']
        image = Image.open('datasets/math_vista/' + image_path)
        D, I = image_to_image(image, model, preprocess, image_index, topk=50)
        new_item = item.copy()
        results = []
        for index, distance in zip(I[0], D[0]):
            result = {}
            result['image_path'] = str(index+1) + ".jpg"
            result['quesiton'] = data.iloc[index]['question']
            result['answer'] = data.iloc[index]['answer']
            choices = data.iloc[index]['options']
            result['metadata'] = {
                "distance": float(distance), 
                "choices": choices.tolist() if choices is not None else [],
                "subject": data.iloc[index]['subject'],
                }
            results.append(result)

        new_item['retrieved_data'] = results
        i2i_results[pid] = new_item

        question = item['question']
        D, I = text_to_text(question, model, text_index, topk=50)
        new_item = item.copy()
        results = []
        for index, distance in zip(I[0], D[0]):
            result = {}
            result['image_path'] = str(index+1) + ".jpg"
            result['quesiton'] = data.iloc[index]['question']
            result['answer'] = data.iloc[index]['answer']
            choices = data.iloc[index]['options']
            result['metadata'] = {
                "distance": float(distance), 
                "choices": choices.tolist() if choices is not None else [],
                "subject": data.iloc[index]['subject'],
                }
            results.append(result)

        new_item['retrieved_data'] = results
        t2t_results[pid] = new_item

    return i2i_results, t2t_results


def retrieve_mathverse(model, preprocess, image_index, text_index, test_dataset):
    """ Retrieve similar images and text from MathVerse dataset.
    Returns the retrieved data.

    Args:
        model (CLIP): CLIP model.
        preprocess (torchvision.transforms): Image preprocessing pipeline.
        image_index (faiss.Index): Faiss index for image embeddings.
        text_index (faiss.Index): Faiss index for text embeddings.
        test_dataset (list): List of test data.
    """

    data = pd.read_parquet('datasets/math_verse/testmini_text_only.parquet')
    i2i_results = {}
    t2t_results = {}

    for pid, item in tqdm(test_dataset.items(),desc="Retrieving from MathVerse"):
        image_path = item['image']
        image = Image.open('datasets/math_vista/' + image_path)
        D, I = image_to_image(image, model, preprocess, image_index, topk=50)

        new_item = item.copy()
        results = []
        for index, distance in zip(I[0], D[0]):
            result = {}
            result['image_path'] = f"image_{index+1}.jpg"
            result['quesiton'] = data.iloc[index]['question']
            result['answer'] = data.iloc[index]['answer']
            result['metadata'] = {
                "distance": float(distance), 
                "query": data.iloc[index]['query_cot'],
                }
            results.append(result)

        new_item['retrieved_data'] = results
        i2i_results[pid] = new_item

        question = item['question']
        D, I = text_to_text(question, model, text_index, topk=50)
        new_item = item.copy()
        results = []
        for index, distance in zip(I[0], D[0]):
            result = {}
            result['image_path'] = f"image_{index+1}.jpg"
            result['quesiton'] = data.iloc[index]['question']
            result['answer'] = data.iloc[index]['answer']
            result['metadata'] = {
                "distance": float(distance), 
                "query": data.iloc[index]['query_cot'],
                }
            results.append(result)

        new_item['retrieved_data'] = results
        t2t_results[pid] = new_item

    return i2i_results, t2t_results


if __name__ == "__main__" :

    test_dataset = []
    with open('datasets/math_vista/test.json', 'r') as f:
        test_dataset = json.load(f)

    # Load the CLIP model
    clip_model, preprocess = clip.load('ViT-L/14@336px', 
                                       device=device, 
                                       jit=False)


    ################################ We Math ###################################
    image_index1 = faiss.read_index('indexes/wemath_image.index')
    text_index1 = faiss.read_index('indexes/wemath_text.index')

    i2i_results, t2t_results = retrieve_wemath(clip_model, preprocess, 
                                                  image_index1, text_index1, 
                                                  test_dataset)

    with open('output/retrieved/mathvista_wemath_i2i.json', 'w') as f:
        json.dump(i2i_results, f, indent=4)
    with open('output/retrieved/mathvista_wemath_t2t.json', 'w') as f:
        json.dump(t2t_results, f, indent=4)


    ################################ Math Vision ###############################
    image_index2 = faiss.read_index('indexes/mathvision_image.index')
    text_index2 = faiss.read_index('indexes/mathvision_text.index')

    i2i_results, t2t_results = retrieve_mathvision(clip_model, preprocess, 
                                                   image_index2, text_index2, 
                                                   test_dataset)

    with open('output/retrieved/mathvista_mathvision_i2i.json', 'w') as f:
        json.dump(i2i_results, f, indent=4)
    with open('output/retrieved/mathvista_mathvision_t2t.json', 'w') as f:
        json.dump(t2t_results, f, indent=4)


    ############################### Math Verse #################################
    image_index3 = faiss.read_index('indexes/mathverse_image.index')
    text_index3 = faiss.read_index('indexes/mathverse_text.index')

    i2i_results, t2t_results = retrieve_mathverse(clip_model, preprocess, 
                                                  image_index3, text_index3, 
                                                  test_dataset)

    with open('output/retrieved/mathvista_mathverse_i2i.json', 'w') as f:
        json.dump(i2i_results, f, indent=4)
    with open('output/retrieved/mathvista_mathverse_t2t.json', 'w') as f:
        json.dump(t2t_results, f, indent=4)