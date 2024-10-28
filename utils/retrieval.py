import torch
import clip

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
        img = preprocess(image).unsqueeze(0).cuda()
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
        text_features = model.encode_text(text_tokens.cuda())

        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embeddings = text_features.cpu().detach().numpy().astype('float32')

        D, I = ind.search(text_embeddings, topk)

    return D, I