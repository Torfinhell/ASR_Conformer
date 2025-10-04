import torch
from src.text_encoder import CTCTextEncoder
import torch.nn.functional as F
def collate_fn(dataset_items:list[dict]):
    assert len(dataset_items)
    batch_by_column={key:[item[key] for item in dataset_items] for key in dataset_items[0].keys()}
    spectorgrams, encoded_texts=batch_by_column["spectorgram"], batch_by_column["text_encoded"]
    max_len_text=max([len(text) for text in encoded_texts])
    new_encoded_texts=[text.tolist()+[0]*(max_len_text-len(text)) for text in encoded_texts]
    text_encoded_lengths=[len(text) for text in encoded_texts]
    max_len_spec=max([spectogram.shape[2] for spectogram in spectorgrams])
    new_spectorgrams=[F.pad(spectorgram, (0,max_len_spec-spectorgram.shape[2],0,0,0,0)).squeeze().tolist() for spectorgram in spectorgrams]
    spectogram_lengths=[spectogram.shape[2] for spectogram in spectorgrams]
    text=[item["text"] for item in dataset_items]
    return  {"spectorgram": torch.tensor(new_spectorgrams),"text_encoded": torch.tensor(new_encoded_texts),
              "text_encoded_lengths":torch.tensor(text_encoded_lengths),
             "spectogram_lengths":torch.tensor(spectogram_lengths),
             "text":text}