# Gymnasium RL molecules
Custom gym environment built for molecular generation using reinforcement learning (RL). It is based on graph representations and uses RDKit functions to analyze the properties and validity of the generated compounds. By default, it proposes candidate drug-molecules with quantitative estimate of drug-likeness (QED) as close as possible to 0.8. The action space is discrete and includes four action types: add node (atom or fragment), add edge (single bond), remove node and stop (terminal state). We use the reward introduced by Noutahi et al. (2024).

## Installation
* To install the environment, please move to the gym-rl-molecules folder:
```bash
$ cd gym-rl-molecules
```
* Next, run the following command:
```bash
$ pip install -e .
```
* The environment can now be used to train the RL model with the provided python code:
```python
import gymnasium as gym
import gym_rl_molecules
env = gym.make('rl-molecules-v0')