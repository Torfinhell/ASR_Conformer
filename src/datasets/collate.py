import torch
from src.text_encoder import CTCTextEncoder
def collate_fn(dataset_items:list[dict]):
    wavs_tensors, encoded_texts=zip(*dataset_items)
    max_len_text=max([len(text) for text in encoded_texts])
    new_encoded_texts=[text.tolist()+[0]*(max_len_text-len(text)) for text in encoded_texts]
    max_len_audio=max([wav.shape for wav in wavs_tensors])
    new_wavs=[wav.tolist()+[0]*(max_len_audio[0]-len(wav)) for wav in wavs_tensors]
    return  torch.Tensor(new_wavs), torch.Tensor(new_encoded_texts)