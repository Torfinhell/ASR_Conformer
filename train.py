import hydra
from src.trainer import BaseTrainer
from src.datasets import collate_fn, BaseDataset
import soundfile as sf
from datasets import load_dataset
import librosa
from IPython import display
import matplotlib.pyplot as plt
import os
import torchaudio
import torch
from torch.utils.data import DataLoader
from hydra.utils import instantiate
@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    text_encoder=instantiate(config.text_encoder)
    librispeech_dataset = BaseDataset(
        text_encoder=text_encoder,
        # shuffle_index=True,
        max_audio_length=config.trainer.max_audio_length,
        config=config,
        part="test"
    )
    dataloader=instantiate(
        config.dataloader,
        dataset=librispeech_dataset, #change spectogram
        collate_fn=collate_fn,
        drop_last=(config.trainer.dataset_partition=="train"),
        shuffle=(config.trainer.dataset_partition=="train"),
        )
    model=instantiate(
        config.model
    )
    optimizer=instantiate(
        config.optimizer,
        params=model.parameters(),
    )
    criterion=instantiate(
        config.loss_function
    )
    trainer=BaseTrainer(
        model=model, 
        criterion=criterion,
        metrics=None, 
        optimizer=optimizer,
        lr_scheduler=None,
        config=config,
        device=config.trainer.device,
        dataloaders=dataloader,
        logger=None,
        writer=None,
        epoch_len=config.trainer.epoch_len,
        n_epochs=config.trainer.n_epochs,
        batch_transforms=None,
    )
    trainer.train()
if __name__=="__main__":
    main()