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
        self.met=metrics


    def train(self):
        self.is_train=True
        self.model=self.model.to(self.device)
        for e in range(self.n_epochs):
            losses=[]
            with tqdm(enumerate(self.dataloaders["train"]), desc=f"Processing epoch {e} in train", total=len(self.dataloaders["train"]), position=0) as pbar_train:
                for idx, batch in pbar_train:
                    self.optimizer.zero_grad()
                    batch=move_to_device(batch, self.device_tensors, self.device)
                    pred=self.model(**batch)
                    batch.update(pred)
                    loss=self.criterion(**batch)
                    batch.update(loss)
                    batch["loss"].backward()
                    self.optimizer.step()
                    losses.append(batch["loss"].item())
                    if(idx%500==0):
                        torch.cuda.empty_cache()
            print(f"Avg Loss in {e} epoch is:{sum(losses)/len(losses)}")
            result={}
            with torch.no_grad():
                 with tqdm(enumerate(self.dataloaders["test"]), desc=f"Processing epoch {e} in test", total=len(self.dataloaders["test"]), position=1, leave=False) as pbar_test:
                    for idx, batch in pbar_test:
                        for met in self.met:
                            batch=move_to_device(batch, self.device_tensors, self.device)
                            pred=self.model(**batch)
                            batch.update(pred)
                            result[f"{met.name}_sum"]=result.get(f"{met.name}_sum",0)+met(**batch)
                            result[f"{met.name}_count"]=result.get(f"{met.name}_count",0)+1
            for met in self.met:
                print(f"Metric {met.name} in {e} epoch is:{result[f"{met.name}_sum"]/result[f"{met.name}_count"]}")
            
def move_to_device(batch, cfg_change_device, device):
    for cfg in cfg_change_device:
        batch[cfg]=batch[cfg].to(device)
    return batch

