# PA2: Deep Q-Networks on MountainCar-v0

This folder contains the submitted MountainCar DQN implementations and experiment scripts provided for Assignment 2.

## Contents

- `dqn_core.py`, `per_buffer.py`, `train.py`: shared DQN, prioritized replay, and training code for the available experiment runners.
- `run_q4a.py`, `run_q4d.py`, `run_q5.py`: replay-factor, sensitivity, and prioritized replay experiments.
- `plot_q4*.py`, `plot_q5.py`, `plot_utils.py`: plotting and analysis helpers.
- `Q3_2_Vanilla_DQN_Implementation/`: standalone vanilla DQN implementation and its README.
- `Q3_3Vanilla_DQN_Truncate/`: modular DQN/truncation implementation and its README.

## Setup

Use either the Conda environment or pip requirements:

```bash
conda env create -f environment.yml
conda activate pa2_dqn
```

or

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the available experiment suite

Run from this directory:

```bash
python run_all.py --num_seeds 2
```

Omit `--num_seeds 2` for the default 15-seed runs. The suite runs the experiment scripts and plotters present in this folder. The standalone implementations have separate setup and run instructions in their own subfolder READMEs.

Generated checkpoints, logs, plots, and videos are not included in this source snapshot.
