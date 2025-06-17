This repository contains the implementation of "_De novo_ Molecular Design through RL-based Generative Models".

## Project overview
This article introduces a RL-based generative model for drug design and optimization. To address the challenges inherent to the large scale of the chemical space, we direct the search with a predefined set of relevant scaffolds, prior knowledge and a chemically significant reward function.
![MDP](https://github.com/user-attachments/assets/7d12e399-d7e0-4c04-b5ba-e3c743d8f22d)
<sub>Overview of the RL loop for molecular design. The agent interacts with the environment and the observed transitions are stored in an experience replay buffer. During the training phase a random batch of experiences is sampled and used to update the model.</sub>

## Code description
The repository is structured around the following key files:
1. `run_training.py` allows to configure and run the training process.
2. `neural_network.py` defines the action-value function and `policy.py` determines how the posterior action selection is made.
3. The molecule design environment is in `gym-rl-molecules/gym_rl_molecules/envs/MoleculesRL.py`.

## Installation
1. Install all required dependencies from `environment.yml` by running:
   ```
   conda env create -f environment.yml
   ```
2. The environment should now be activated with:
   ```
   conda activate Mol_RL
   ```
3. Install customized RL environment for molecule generation:
   ```
   cd gym-rl-molecules
   pip install -e .
   ```

## Usage

### Model training
#### General Considerations
The model can be used for both *de novo* molecular design and the optimization of existing drug molecules. Each tasks is defined through the following command-line arguments:

| Argument       | *De novo*      | Optimization   |
|----------------|----------------|----------------|
| `--max_episode_steps`  | 16  |   1  |
| `--qed_target`  | 0.8  | 1 |
| `--initial_scaffold`  | carbon  | moses |

#### Run

1. *(Optional, recommended)* Log in to Weights & Biases using your API key to track the training process:
   ```
   wandb login <API_key>
   ```
2. Train the model:
   ```
   python run_training.py
   ```
   🔹 **Important**: if training is made in a SLURM environment, add `srun` at the beginning of the command to guarantee proper parallelization.

   🔹 Add any desired command-line arguments to customize the training. If Weights & Biases is used, add `--use_wandb=True`. For a detailed explanation of the options, please use the following command:
      ```
       python run_training.py --help
      ```

### Pool generation

Generate a pool of molecules from a model checkpoint.  An example of checkpoint for *de novo* molecular design is provided at `pool/example_checkpoint.pth`.
```
python pool/generate_pool.py
```
   🔹 **Important**: if execution is made in a SLURM environment, add `srun` at the beginning of the command to guarantee proper parallelization.
   🔹 For a detailed explanation of the command-line arguments, please use the following command:
      ```
      python pool/generate_pool.py --help
      ```

### Synthetic experience creation

The code for generating synthetic experiences is provided at `synthetic_experiences/synthetic_experiences.py`. To run it, use the following command:
```
python synthetic_experiences/synthetic_experiences.py
```
   🔹 **Important**: if execution is made in a SLURM environment, add `srun` at the beginning of the command to guarantee proper parallelization.
   🔹 For a detailed explanation of the command-line arguments, please use the following command:
      ```
      python synthetic_experiences/synthetic_experiences.py --help
      ```
