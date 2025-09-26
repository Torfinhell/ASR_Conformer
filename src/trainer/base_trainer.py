import torch
class BaseTrainer:
    """
    Base class for all trainers.
    """
    def __init__(
            self,
            model, 
            criterion,
            metrics, 
            optimizers,
            lr_scheduler,
            config,
            device,
            dataloaders,
            logger,
            writer,
            epoch_len=None,
            skip_oom=None,
            batch_transforms=None,
    ):
        self.is_train=True
        self.config=config
    def train(self):
        pass