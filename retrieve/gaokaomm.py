import clip
import torch
import numpy as np
from tqdm import tqdm
import faiss
from glob import glob
import os
import json
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_ids = list(range(torch.cuda.device_count()))


def numpy_float32_default(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            raise TypeError


def build_faiss_image(val_dataset, device, model, preprocess=None):
    embeddings = []
    index_to_image_id = {}
    count = 0
    for datum in tqdm(val_dataset):
        pics = datum['picture']

        for j in range(len(pics)):
            # image_id is the image path
            # e.g. "../Data/2010-2023_Biology_MCQs/2010-2023_Biology_MCQs_0_0.png"
            image_id = pics[j]
            if image_id in index_to_image_id.values():
                continue
            image_path = 'datasets/GaokaoMM/Data/' + pics[j]

        with torch.no_grad():
            image = preprocess(Image.open(image_path)).to(device)
            image_embeddings = model.encode_image(torch.unsqueeze(image, dim=0))

        combined_embedding = image_embeddings
        normalized_embedding = combined_embedding / combined_embedding.norm(
            dim=-1, keepdim=True
        )
        embeddings.append(normalized_embedding.cpu().numpy())

        index_to_image_id[count] = image_id
        count += 1

    embeddings = np.vstack(embeddings).astype('float32')

    # cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    return index, index_to_image_id


def build_faiss_text(val_dataset, device, model):
    embeddings = []
    index_to_text_id = {}
    count = 0

    for datum in tqdm(val_dataset):
        question = datum['question']
        # text_id is the file name + index
        # e.g. "2010-2023_Biology_MCQs+0"
        text_id = datum['picture'][0].split('/')[2] + '+' + str(datum['index'])

        with torch.no_grad():
            text_tokens = clip.tokenize([question], truncate=True)
            text_embedding = model.encode_text(text_tokens.to(device))
        
        combined_embedding = text_embedding
        normalized_embedding = combined_embedding / combined_embedding.norm(
            dim=-1, keepdim=True
        )
        embeddings.append(normalized_embedding.cpu().numpy())

        index_to_text_id[count] = text_id
        count += 1

    embeddings = np.vstack(embeddings).astype('float32')

    # cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    return index, index_to_text_id


def load_faiss_index(index_path, index_to_image_id_path):
    index = faiss.read_index(index_path)
    with open(index_to_image_id_path, 'r', encoding='utf-8') as f:
        index_to_image_id = json.load(f)
    return index, index_to_image_id


def text_to_image(text, model, ind, topk=5):
    with torch.no_grad():
        text_tokens = clip.tokenize([text], truncate=True)
        text_features = model.encode_text(text_tokens.to(device))

        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embeddings = text_features.cpu().detach().numpy().astype('float32')

        D, I = ind.search(text_embeddings, topk)
    return D, I


def image_to_image(image, model, preprocess, ind, topk=5):
    with torch.no_grad():
        img = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(img)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_embeddings = image_features.cpu().detach().numpy().astype('float32')
        
        D, I = ind.search(image_embeddings, topk)
    
    return D, I


def text_to_text(text, model, ind, topk=5):
    with torch.no_grad():
        text_tokens = clip.tokenize([text], truncate=True)
        text_features = model.encode_text(text_tokens.to(device))

        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embeddings = text_features.cpu().detach().numpy().astype('float32')

        D, I = ind.search(text_embeddings, topk)
    return D, I 


if __name__ == '__main__':

    val_dataset = []
    directory = 'datasets/GaokaoMM/'
    json_files = glob(os.path.join(directory, '*.json'))

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file).get('example')
            val_dataset += data
    val_dataset = [item for item in val_dataset if item.get('year') not in ['2023', '2022']]
    clip_model, preprocess = clip.load('ViT-L/14@336px', device='cuda', jit=False)

    img_index, index_to_image_id = build_faiss_image(
        val_dataset,
        device,
        clip_model,
        preprocess
    )

    txt_index, index_to_text_id = build_faiss_text(
        val_dataset,
        device,
        clip_model
    )

    # faiss.write_index(
    #     img_index,
    #     'datasets/faiss_index/GaokaoMM_test_image.index'
    # )

    # with open('index_to_image_id.json', 'w', encoding='utf-8') as f:
    #     json.dump(index_to_image_id, f, ensure_ascii=False, indent=4)

    # img_index, index_to_image_id = load_faiss_index(
    #     'datasets/faiss_index/GaokaoMM_test_image.index',
    #     'index_to_image_id.json'
    # )

    for json_file in tqdm(json_files):
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            keywords = data.get('keywords')
            example = data.get('example')
            example = [item for item in example if item.get('year') in ['2022', '2023']]

        i2i_output_data = {
            'keywords': keywords,
            'example': [],
            }
        t2t_output_data = {
            'keywords': keywords,
            'example': [],
            }

        for item in example:

            question = item['question']
            images = item['picture']

            i2i_result_entry = {
                'year': item['year'],
                'category': item['category'],
                'question': question,
                'picture': images,
                'answer': item['answer'],
                'analysis': item['analysis'],
                'index': item['index'],
                'score': item['score'],
                'retrieved_results': []
            }
            t2t_result_entry = {
                'year': item['year'],
                'category': item['category'],
                'question': question,
                'picture': images,
                'answer': item['answer'],
                'analysis': item['analysis'],
                'index': item['index'],
                'score': item['score'],
                'retrieved_results': []
            }

            # Retrieve text
            D,I = text_to_text(question, clip_model, txt_index)
            for idx, distance in zip(I[0], D[0]):
                print(index_to_text_id[idx])
                rtext_filename = index_to_text_id[idx].split('+')[0]
                rtext_id = index_to_text_id[idx].split('+')[1]

                with open('datasets/GaokaoMM/'+rtext_filename+'.json', 
                              'r', encoding='utf-8') as file:
                        link_data = json.load(file)

                retrieved_txt = None
                rpicture = None
                ranswer = None
                ranaysis = None

                # link back to the source quesion
                for item in link_data.get('example'):
                    if item['index'] == int(rtext_id):
                        retrieved_txt = item['question']
                        rpicture = item['picture']
                        ranswer = item['answer']
                        ranaysis = item['analysis']

                t2t_result_entry['retrieved_results'].append({
                    'retrieved_txt': retrieved_txt,
                    'distance': distance,
                    'picture': rpicture,
                    'answer': ranswer,
                    'anaysis': ranaysis,
                })
            
            t2t_output_data['example'].append(t2t_result_entry)
                
            
            # Retrieve image
            for image in images:
                img = Image.open('datasets/GaokaoMM/Data/'+image)
                D, I = image_to_image(img, clip_model, preprocess, 
                                      img_index, topk=5)

                img_retrieved = []
                for idx, distance in zip(I[0], D[0]):
                    retrieved_img = index_to_image_id[idx]
                    
                    # link back to the source question
                    rquestion = None
                    ranswer = None
                    ranaysis= None

                    filename = retrieved_img.split('/')[2]
                    with open('datasets/GaokaoMM/'+filename+'.json', 
                              'r', encoding='utf-8') as file:
                        link_data = json.load(file)
                    for item in link_data['example']:
                        if retrieved_img in item['picture']:
                            rquestion = item['question']
                            ranswer = item['answer']
                            ranaysis = item['analysis']
                    
                    img_retrieved.append({
                        'retrieved_img': retrieved_img,
                        'distance': distance,
                        'question': rquestion,
                        'answer': ranswer,
                        'analysis':ranaysis,
                    })

                i2i_result_entry['retrieved_results'].append(img_retrieved)

            i2i_output_data['example'].append(i2i_result_entry)
            
        
        with open('output/t2t/' + json_file.split('/')[-1], 'w', encoding='utf-8') as f:
            json.dump(t2t_output_data, f, default=numpy_float32_default, 
                      ensure_ascii=False, indent=4)
        with open('output/i2i/' + json_file.split('/')[-1], 'w', encoding='utf-8') as f:
            json.dump(i2i_output_data, f, default=numpy_float32_default, 
                      ensure_ascii=False, indent=4)
            
        

