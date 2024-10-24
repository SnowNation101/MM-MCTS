"""Builds the index for image and text retrieval using CLIP and Faiss."""


import faiss
import torch
import clip
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_ids = list(range(torch.cuda.device_count()))

class IndexBuilder:
    def __init__(self, dataset, device, model, preprocess=None):
        self.dataset = dataset
        self.device = device
        self.model = model
        self.preprocess = preprocess

        dataset_paths = {
            "MathVista": "/home/u2024001042/huggingface/datasets/math_vista/",
            "MathVision": "/home/u2024001042/huggingface/datasets/math_vision/",
            "MathVerse": "/home/u2024001042/huggingface/datasets/math_verse/"
        }

        self.path = dataset_paths.get(self.dataset)

    def build_image_index(self):
        embeddings = []
        index_to_image_id = {}
        count = 0

        if self.dataset == "MathVista":
            image_dir = os.path.join(self.path, 'images')

        for i in tqdm(range(1, 1001)):
            image_file = f"{i}.jpg"
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

        for _, datum in tqdm(data.iterrows(), total=len(data)):
            question = datum['question']
            text_id = datum['pid']

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


        index.add(embeddings)



    def load_index(self, index_path, index_to_id_path):
        # Your implementation for loading the Faiss index
        # Return the loaded index and index mapping
        pass

# Usage in your main script
if __name__ == '__main__':
    clip_model, preprocess = clip.load('ViT-L/14@336px', device='cuda', jit=False)

    # Initialize the IndexBuilder with dataset, device, model, preprocess
    builder = IndexBuilder("MathVista", device, clip_model, preprocess)

    # Build image index
    # img_index = builder.build_image_index()
    text_index = builder.build_text_index()

    # Build text index
    # txt_index = builder.build_text_index()

    # Saving and loading the index can be done as follows
    # img_index = builder.load_index('image_index_path', 'image_index_to_id_path')

    # txt_index = builder.load_index('text_index_path', 'text_index_to_id_path')

    # The rest of your code for processing the examples can remain in the main script