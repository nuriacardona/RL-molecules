# _De novo_ Molecular Design through RL-based Generative Models

This repository contains the implementation of "_De novo_ Molecular Design through RL-based Generative Models".

## Project overview
This study introduces a RL-based generative model for drug design and optimization.

## Code description
This repository is structured around the following key files:
1. `run_training.py` is the file to configure and run the training process.
2. `neural_network.py` defines the action-value function and `policy.py` determines how the posterior action selection is made.
3. The molecule design environment is in `gym-rl-molecules/gym_rl_molecules/envs/MoleculesRL.py`.
