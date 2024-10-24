""" Use We-Math as questions to retrieve the answer from MathVerse, MathVista, 
and MathVision.
"""

import clip
import torch
import faiss
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_ids = list(range(torch.cuda.device_count()))

def image_to_image(image, model, preprocess, ind, topk=5):
    """ Retrieves similar images from the index using the image embeddings.
    Returns the distance and index of the topk similar images.

    Args:
        image (PIL.Image): Image to retrieve similar images.
        model (CLIP): CLIP model.
        preprocess (torchvision.transforms): Image preprocessing pipeline.
        ind (faiss.Index): Faiss index.
        topk (int): Number of similar images to retrieve.
    """

    with torch.no_grad():
        img = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(img)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_embeddings = image_features.cpu().detach().numpy().astype('float32')
        
        D, I = ind.search(image_embeddings, topk)
    
    return D, I


def text_to_text(text, model, ind, topk=5):
    """ Retrieves similar text from the index using the text embeddings.
    Returns the distance and index of the topk similar text.
    
    Args:  
        text (str): Text to retrieve similar text.
        model (CLIP): CLIP model.
        ind (faiss.Index): Faiss index.
        topk (int): Number of similar text to retrieve.
    """

    with torch.no_grad():
        text_tokens = clip.tokenize([text], truncate=True)
        text_features = model.encode_text(text_tokens.to(device))

        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embeddings = text_features.cpu().detach().numpy().astype('float32')

        D, I = ind.search(text_embeddings, topk)

    return D, I 


if __name__ == "__main__" :

    test_dataset = []
    with open('/home/u2024001042/huggingface/datasets/we_math/testmini.json', 'r') as f:
        test_dataset = json.load(f)

    # Load the CLIP model
    clip_model, preprocess = clip.load('ViT-L/14@336px', device=device, jit=False)


    image_index = faiss.read_index('indexes/mathvista_image.index')
    text_index = faiss.read_index('indexes/mathvista_text.index')

    i2i_results = []
    t2t_results = []

    data = pd.read_parquet('datasets/math_vista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet')

    for item in tqdm(test_dataset):
        image_path = item['image_path']
        image = Image.open('datasets/we_math/'+image_path)
        D, I = image_to_image(image, clip_model, preprocess, image_index, topk=50)

        new_item = item.copy()
        results = []
        for idx, distance in zip(I[0], D[0]):
            result = {}
            result['image_path'] = str(idx) + ".jpg"
            result['quesiton'] = data.iloc[int(idx)]['question']
            result['answer'] = data.iloc[int(idx)]['answer']
            choices = data.iloc[int(idx)]['choices']
            result['metadata'] = {
                "distance": float(distance), 
                "choices": choices.tolist() if choices is not None else [],
                "query": data.iloc[int(idx)]['query'],
                }
            results.append(result)

        new_item['retrieved_data'] = results
        i2i_results.append(new_item)

        question = item['question']
        D, I = text_to_text(question, clip_model, text_index, topk=50)
        new_item = item.copy()
        results = []
        for idx, distance in zip(I[0], D[0]):
            result = {}
            result['image_path'] = str(idx) + ".jpg"
            result['quesiton'] = data.iloc[int(idx)]['question']
            result['answer'] = data.iloc[int(idx)]['answer']
            choices = data.iloc[int(idx)]['choices']
            result['metadata'] = {
                "distance": float(distance), 
                "choices": choices.tolist() if choices is not None else [],
                "query": data.iloc[int(idx)]['query'],
                }
            results.append(result)

        new_item['retrieved_data'] = results
        t2t_results.append(new_item)

        
    
    with open('output/retrieved/wemath_mathvista_i2i.json', 'w') as f:
        json.dump(i2i_results, f, indent=4)

    with open('output/retrieved/wemath_mathvista_t2t.json', 'w') as f:
        json.dump(t2t_results, f, indent=4)



