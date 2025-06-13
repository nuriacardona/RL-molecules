from gymnasium.envs.registration import register

register(
    id='rl-molecules-v0',
    entry_point='gym_rl_molecules.envs:MolecEnv',
)