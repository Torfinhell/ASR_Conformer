import torchaudio
import os
from torch.utils.data import Dataset
import torchaudio.functional as F
from hydra.utils import instantiate
class BaseDataset(Dataset):
    def __init__(
            self, 
            url="test-clean",
            text_encoder=None,
            max_text_length=None,
            max_audio_length=None,
            config=None,
            part="train"
            ):
        #sample rate change
        self.text_encoder=text_encoder
        self.base_sample_rate=config.trainer.base_sample_rate if config is not None else 16000
        self.dataset = instantiate(
            config.datasets.train, 
            root=os.path.expanduser("~/.cache"),
            download=True,
        )  
        self.dataset=filter_records_by_length(self.dataset)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        sr=self.dataset[index][1]
        wav_tensor=self.dataset[index][0]
        if(sr!=self.base_sample_rate):
            wav_tensor=F.resample(wav_tensor, sr, self.base_sample_rate, lowpass_filter_width=6)
        text=self.dataset[index][2]
        text=self.text_encoder.encode(text)
        return wav_tensor.squeeze(0), text.squeeze(0)


def filter_records_by_length(dataset):
    return dataset
    
        