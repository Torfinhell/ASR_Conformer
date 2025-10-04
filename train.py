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
from tqdm import tqdm
@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    text_encoder=instantiate(config.text_encoder)
    dataloaders={}
    for part in {"train", "test"}:
        dataset= BaseDataset(
            text_encoder=text_encoder,
            max_audio_length=config.trainer.max_audio_length,
            config=config,
            part=part
        )
        dataloaders[part]=instantiate(
            config.dataloader,
            dataset=dataset, 
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
    metrics=instantiate(
        config.metrics
    )
    trainer=BaseTrainer(
        model=model, 
        criterion=criterion,
        metrics=metrics, 
        optimizer=optimizer,
        lr_scheduler=None,
        config=config,
        device=config.trainer.device,
        dataloaders=dataloaders,
        logger=None,
        writer=None,
        epoch_len=config.trainer.epoch_len,
        n_epochs=config.trainer.n_epochs,
        batch_transforms=None,
    )
    trainer.train()
if __name__=="__main__":
    main()
