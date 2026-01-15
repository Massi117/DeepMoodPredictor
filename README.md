# DeepMoodPredictor

## Overview

DeepMoodPredictor is a PyTorch-based library for predicting mood states from functional magnetic reasonance data (fMRI). It provides configurable model architectures, preprocessing utilities, training and evaluation pipelines, and scripts for reproducible experiments.

## Key Features

- Modular model components (RNN, CNN, Transformer blocks)
- Flexible preprocessing and augmentation for physiological/sensor streams
- Config-driven training and evaluation
- Checkpointing, logging, and metric tracking
- Example notebooks and demo scripts

## Requirements

- Python 3.10.14
- PyTorch 2.5.1
- Common packages: numpy, pandas, scikit-learn, matplotlib, tqdm, hydra-core (recommended)

## Installation

1. Create and activate a virtual environment:
    python -m venv .venv
    source .venv/bin/activate

2. Install dependencies:
    pip install -r requirements.txt

3. (Optional) Install in editable mode for development:
    pip install -e .


## License

Distributed under the MIT License. See LICENSE for details.

## Contact

For questions or collaboration, open an issue or contact the maintainer via the repository GitHub page.