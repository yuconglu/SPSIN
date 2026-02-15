# SPSIN

## 1. Environment Setup

The project is developed with Python 3.11 and PyTorch 2.4.0 (for CUDA 12.8).

The main dependencies are:
- numpy==1.26.4
- pandas==2.2.2
- torch==2.4.0
- torchvision==0.19.0
- torch-geometric==2.5.3
- torchdiffeq==0.2.3
- optuna==3.6.1
- timm==0.9.12


## 2. Dataset Download

The model is trained and evaluated on the ERA5 reanalysis dataset, which is publicly available.

You can access and download the WeatherBench data from:
[**WeatherBench Dataset**](https://mediatum.ub.tum.de/1524895)

optimization study by repeatedly calling the `run_training_job` function from `Run.py`.

