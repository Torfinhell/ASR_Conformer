# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a solution for solving ASR task with PyTorch

See the task assignment [here](https://github.com/markovka17/dla/tree/2025/hw1_asr).

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

   a. `conda` version:

   ```bash
   # create env
   conda create -n hifi_gan python=3.11
   # activate env
   conda activate hifi_gan
   ```

1. Install all required packages

   ```bash
   pip install uv
   uv sync
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use
To download models checkpoints and test dataset run following:
```
!uv run scripts/download_gdrive.py
```

## How To Use

To train best model, run the following commands:

```bash
uv run train.py HYDRA_CONFIG_ARGUMENTS -cn=train_clean_360_1
```
and then to finetune afterwards:
```bash
uv run train.py +trainer.from_pretrained=PREV_CHECKPOINT OTHER_HYDRA_CONFIG_ARGUMENTS -cn=train_other_500_2
```
## How To Inference and Evaluate
To evaluate the model run:
```bash
!uv run inference.py \
   inferencer.from_pretrained={model_path} text_encoder=CTCEncoder \
   inferencer.save_path={output_dir} text_encoder.beam_size=100 \
   -cn=inference_all_metrics
```
To save predictions run:
```bash
!uv run inference.py dataloader=onebatchtest \
            inferencer.dataset_dir={dataset_dir} \
            inferencer.from_pretrained={model_path} \
            inferencer.save_path={gt_name} \
            text_encoder=CTCEncoder -cn=inference

```


## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
