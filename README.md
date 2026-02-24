# Coding Historical Causes of Death with Large Language Models

This project investigates whether modern Large Language Models (LLMs) can automatically translate historical free-text descriptions of causes of death into ICD10h codes.

Historical demographers currently perform this coding manually. The process is time-consuming, requires specialised knowledge, and does not scale to rapidly growing digitised archives. This project aims to reduce coding time from months to minutes using machine learning.

## Project Goals

- Develop a training pipeline for fine-tuning an LLM on historical cause-of-death data.
- Evaluate model accuracy and robustness across cities, languages, and time periods.
- Build a simple prototype tool for applying the model to new datasets, including low-resource settings.

## Data

This project builds on existing European historical datasets, including manually coded cause-of-death records with ICD10h classifications.

> **Important:** The datasets used in this project are not included in this repository and are subject to separate data-sharing agreements with the respective institutions.

## Method Overview

The pipeline consists of:

1. Data preprocessing and normalization
2. Fine-tuning of a transformer-based language model
3. Cross-city and cross-period evaluation
4. Robustness analysis
5. Prototype inference interface

## Repository Structure

```
codLLM/
├── main.py                         # Main entry point
├── pyproject.toml                  # Project metadata and dependencies
├── data/                           # Data directory (no raw data included)
├── dockerfiles/
│   └── train.dockerfile
├── models/
│   └── placeholder.pth
├── src/
│   └── codllm/
│       ├── config.py               # Project configuration
│       ├── data_augmentation.py    # Data augmentation utilities
│       ├── data_handler.py         # Dataset loading and mapping
│       ├── model_registry.py       # Model loader registry
│       ├── preprocess.py           # Tokenization preprocessing
│       └── train.py                # Training scripts
└── tests/
    └── test_training.py            # Training tests
```

## Installation

```bash
git clone https://github.com/your-org/historical-cod-llm
cd historical-cod-llm
uv sync
```

## Docker

Build the training image:

```bash
docker build -f dockerfiles/train.dockerfile -t codllm-train:latest .
```

Run the container:

```bash
docker run --rm -e HUGGINGFACE_HUB_TOKEN -e WANDB_API_KEY codllm-train:latest
```

`WANDB_API_KEY` is optional. When omitted, training runs without Weights & Biases logging in unauthenticated containers.

## License

This repository is licensed under the MIT License.

Note that the historical datasets used for training and evaluation are not publicly available and require separate agreements with the data providers.
