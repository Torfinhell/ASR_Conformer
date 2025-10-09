import torch
from src.text_encoder import CTCTextEncoder
import torch.nn.functional as F


def collate_fn(dataset_items:list[dict]):
    assert len(dataset_items)
    batch_by_column={key:[item[key] for item in dataset_items] for key in dataset_items[0].keys()}
    spectrograms, encoded_texts=batch_by_column["spectrogram"], batch_by_column["text_encoded"]
    max_len_text=max([len(text) for text in encoded_texts])
    new_encoded_texts=[text.tolist()+[0]*(max_len_text-len(text)) for text in encoded_texts]
    text_encoded_lengths=[len(text) for text in encoded_texts]
    max_len_spec=max([spectrogram.shape[2] for spectrogram in spectrograms])
    new_spectrograms=[F.pad(spectrogram, (0,max_len_spec-spectrogram.shape[2],0,0,0,0)).squeeze().tolist() for spectrogram in spectrograms]
    spectrogram_lengths=[spectrogram.shape[2] for spectrogram in spectrograms]
    text=[item["text"] for item in dataset_items]
    return  {"spectrogram": torch.tensor(new_spectrograms),"text_encoded": torch.tensor(new_encoded_texts),
              "text_encoded_lengths":torch.tensor(text_encoded_lengths),
             "spectrogram_lengths":torch.tensor(spectrogram_lengths),
             "text":text,
             "audio_path":batch_by_column["audio_path"]}
