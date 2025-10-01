import torch
from tqdm import tqdm
class BaseTrainer:
    """
    Base class for all trainers.
    """
    def __init__(
            self,
            model, 
            criterion,
            metrics, 
            optimizer,
            lr_scheduler,
            config,
            device,
            dataloaders,
            logger,
            writer,
            epoch_len=None,
            n_epochs=None,
            skip_oom=None,
            batch_transforms=None,
    ):
        self.is_train=True
        self.n_epochs=n_epochs
        self.epoch_len=config.trainer.epoch_len
        self.model=model
        self.optimizer=optimizer
        self.lr_scheduler=lr_scheduler
        self.device=device
        self.dataloaders=dataloaders
        self.logger=logger
        self.is_train=True
        self.criterion=criterion
        self.device_tensors=config.trainer.device_tensors



    def train(self):
        self.is_train=True
        self.model=self.model.to(self.device)
        for e in range(self.n_epochs):
            losses=[]
            for idx, batch in tqdm(enumerate(self.dataloaders), desc=f"Processin epoch {e}", total=len(self.dataloaders)):
                self.optimizer.zero_grad()
                batch=move_to_device(batch, self.device_tensors, self.device)
                pred=self.model(**batch)
                batch.update(pred)
                loss=self.criterion(**batch)
                batch.update(loss)
                batch["loss"].backward()
                self.optimizer.step()
                losses.append(batch["loss"].item())
            print(f"Avg Loss in {e} epoch is:{sum(losses)/len(losses)}")
def move_to_device(batch, cfg_change_device, device):
    for cfg in cfg_change_device:
        batch[cfg]=batch[cfg].to(device)
    return batch

