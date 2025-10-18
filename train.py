import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders, get_texts_for_bpe
from src.trainer import Trainer
from src.transforms.utils import show_augs
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)
    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)
    if config.trainer.get("show_augs"):
        show_augs(config, writer)
        return
    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device
    texts = get_texts_for_bpe()
    # setup text_encoder
    llm_model_config=config.text_encoder.get("llm_model")
    llm_model=instantiate(llm_model_config) if llm_model_config is not None else None
    text_encoder = instantiate(config.text_encoder, texts=texts, llm_model=llm_model)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)

    # build model architecture, then print to console
    model = instantiate(config.model, n_tokens=len(text_encoder), 
        unfreeze_last_layers=config.trainer.get("unfreeze_last_layers")).to(device)
    # logger.info(model) TODO

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)

    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            # use text_encoder in metrics
            metrics[metric_type].append(
                instantiate(metric_config, text_encoder=text_encoder)
            )

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    if (epoch_len := config.trainer.get("epoch_len")) is None:
        epoch_len = len(dataloaders["train"])
    grad_acum=config.trainer.get("grad_acum")
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer, 
        steps_per_epoch=max(epoch_len//grad_acum, 1)
    )
    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        text_encoder=text_encoder,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
