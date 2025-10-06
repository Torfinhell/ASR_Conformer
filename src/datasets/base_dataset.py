import torchaudio
import os
from torch.utils.data import Dataset
import torchaudio.functional as F
from hydra.utils import instantiate
import random as rng
class BaseDataset(Dataset):
    def __init__(
            self, 
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
            config.datasets[part], 
            root=os.path.expanduser("~/.cache"),
            download=True,
        )  
        self.n_mels=config.model.model_dim
        self.n_fft=config.trainer.n_fft
        self.hop_length=config.trainer.hop_length
        self.dataset=filter_records_by_length(self.dataset)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        sr=self.dataset[index][1]
        wav_tensor=self.dataset[index][0]
        assert sr==self.base_sample_rate
        if(sr!=self.base_sample_rate):
            wav_tensor=F.resample(wav_tensor, sr, self.base_sample_rate, lowpass_filter_width=6)
        text=self.dataset[index][2]
        text_encoded=self.text_encoder.encode(text)
        spectorgram=torchaudio.transforms.MelSpectrogram(sample_rate=self.base_sample_rate, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)(wav_tensor)
        return {"audio":wav_tensor.squeeze(0), 
                "text_encoded":text_encoded.squeeze(0),
                "spectorgram":spectorgram,
                "spectorgram_length":len(spectorgram),
                "text":text
                }
        
def filter_records_by_length(dataset): #change
    return dataset
    
        