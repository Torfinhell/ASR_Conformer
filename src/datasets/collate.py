import torch
import torch.nn.functional as F

from src.text_encoder import CTCTextEncoder


def pad_tensors(tensors: list[torch.Tensor]):
    """
    pad tensors with same number of dimensiona to the same shape
    """
    max_shape = [max(s) for s in zip(*[t.shape for t in tensors])]
    padded_tensors = [
        F.pad(
            t,
            [
                (0, max_shape[dim_ind] - t.shape[dim_ind])
                for dim_ind in range(len(max_shape))
            ],
            "constant",
            0,
        )
        for t in tensors
    ]
    return torch.stack(padded_tensors)


def collate_fn(dataset_items: list[dict]):
    assert len(dataset_items)
    batch_by_column = {
        key: [item[key] for item in dataset_items] for key in dataset_items[0].keys()
    }
    result_batch = {
        "spectrogram": pad_tensors(batch_by_column["spectrogram"]),
        "text": pad_tensors(batch_by_column["text_encoded"]),
        "audio_path": batch_by_column["audio_path"],
        "audio": batch_by_column["audio"],
    }
    result_batch.update(
        {
            "text_encoded_lengths": torch.tensor(
                [text.shape[0] for text in result_batch["text"]]
            ),
            "spectrogram_lengths": torch.tensor(
                [spec.shape[2] for spec in result_batch["spectrogram"]]
            ),
        }
    )
    return result_batch
