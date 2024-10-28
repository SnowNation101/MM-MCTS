"""Builds the index for image and text retrieval using CLIP and Faiss."""


import faiss
import torch
import clip
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_ids = list(range(torch.cuda.device_count()))

class IndexBuilder:
    def __init__(self, dataset: str, device, model, preprocess=None):
        self.dataset = dataset
        self.device = device
        self.model = model
        self.preprocess = preprocess

        dataset_paths = {
            "MathVista": "datasets/math_vista/",
            "MathVision": "datasets/math_vision/",
            "MathVerse": "datasets/math_verse/",
            "WeMath": "datasets/we_math/",
        }

        self.path = dataset_paths.get(self.dataset)

    def build_image_index(self):
        embeddings = []
        index_to_image_id = {}
        count = 0

        image_dir = os.path.join(self.path, 'images')
        if self.dataset == "MathVista":
            data_range = range(1, 1001)
        elif self.dataset == "MathVision":
            data_range = range(1, 3041)
        elif self.dataset == "MathVerse":
            image_dir = os.path.join(self.path, 'images/images_version_6')
            data_range = range(1, 789)
        elif self.dataset == "WeMath":
            data_range = range(1740)

        for i in tqdm(data_range):
            if self.dataset == "WeMath":
                with open(os.path.join(self.path, 'testmini.json')) as f:
                    data = json.load(f)
                image_path = os.path.join(self.path,data[i]['image_path'])
                image_id = data[i]['ID']

            else:
                image_file = f"{i}.jpg"
                if self.dataset == "MathVerse":
                    image_file = f"image_{i}.png"
                image_path = os.path.join(image_dir, image_file)

                # Check if the file exists in the directory
                if not os.path.isfile(image_path):
                    continue

                # Use the image name (without .jpg) as the image_id
                image_id = str(i)

            if image_id in index_to_image_id.values():
                continue

            with torch.no_grad():
                image = self.preprocess(Image.open(image_path)).to(device)
                image_embeddings = self.model.encode_image(torch.unsqueeze(image, dim=0))

            combined_embedding = image_embeddings
            normalized_embedding = combined_embedding / combined_embedding.norm(
                dim=-1, keepdim=True
            )
            embeddings.append(normalized_embedding.cpu().numpy())

            # Map the current index to the image_id
            index_to_image_id[count] = image_id
            count += 1

        embeddings = np.vstack(embeddings).astype('float32')

        # cosine similarity
        index = faiss.IndexFlatIP(embeddings.shape[1])

        # Wrap it with IndexIDMap
        index_with_ids = faiss.IndexIDMap(index)

        # Add embeddings with corresponding IDs
        ids = np.array(list(index_to_image_id.keys())).astype('int64')
        index_with_ids.add_with_ids(embeddings, ids)
        faiss.write_index(index_with_ids, 'indexes/' + self.dataset.lower() + '_image.index')

        return index_with_ids


    def build_text_index(self):
        embeddings = []
        index_to_text_id = {}
        count = 0
        
        if self.dataset == "MathVista":
            data = pd.read_parquet(os.path.join(self.path, 'data/testmini-00000-of-00001-725687bf7a18d64b.parquet'))
        elif self.dataset == "MathVision":
            data = pd.read_parquet(os.path.join(self.path, 'data/test-00000-of-00001-3532b8d3f1b4047a.parquet'))
        elif self.dataset == "MathVerse":
            data = pd.read_parquet(os.path.join(self.path, 'testmini_text_only.parquet'))

        if self.dataset == "WeMath":
            for i in tqdm(range(1740)):
                with open(os.path.join(self.path, 'testmini.json')) as f:
                    data = json.load(f)
                question = data[i]['question']
                text_id = data[i]['ID']
                with torch.no_grad():
                    text_tokens = clip.tokenize([question], truncate=True)
                    text_embedding = self.model.encode_text(text_tokens.to(device))
                
                combined_embedding = text_embedding
                normalized_embedding = combined_embedding / combined_embedding.norm(
                    dim=-1, keepdim=True
                )
                embeddings.append(normalized_embedding.cpu().numpy())

                index_to_text_id[count] = text_id
                count += 1
        else:
            for _, datum in tqdm(data.iterrows(), total=len(data)):
                question = datum['question']
                if self.dataset == "MathVista":
                    text_id = datum['pid']
                elif self.dataset == "MathVision":
                    text_id = datum['id']
                elif self.dataset == "MathVerse":
                    text_id = datum['problem_index']

                with torch.no_grad():
                    text_tokens = clip.tokenize([question], truncate=True)
                    text_embedding = self.model.encode_text(text_tokens.to(device))
                
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

        # Wrap it with IndexIDMap
        index_with_ids = faiss.IndexIDMap(index)

        # Add embeddings with corresponding IDs
        ids = np.array(list(index_to_text_id.keys())).astype('int64')
        index_with_ids.add_with_ids(embeddings, ids)

        faiss.write_index(index_with_ids, 'indexes/' + self.dataset.lower() + '_text.index')
        
        return index_with_ids



    def load_index(self, index_path, index_to_id_path):
        # TODO: Load the index and index_to_id mapping
        pass


if __name__ == '__main__':
    clip_model, preprocess = clip.load('ViT-L/14@336px', device='cuda', jit=False)

    builder1 = IndexBuilder("MathVista", device, clip_model, preprocess)
    img_index1 = builder1.build_image_index()
    txt_index1 = builder1.build_text_index()

    builder2 = IndexBuilder("MathVision", device, clip_model, preprocess)
    img_index2 = builder2.build_image_index()
    txt_index2 = builder2.build_text_index()

    builder3 = IndexBuilder("MathVerse", device, clip_model, preprocess)
    img_index3 = builder3.build_image_index()
    txt_index3 = builder3.build_text_index()

    builder4 = IndexBuilder("WeMath", device, clip_model, preprocess)
    img_index4 = builder4.build_image_index()
    txt_index4 = builder4.build_text_index()