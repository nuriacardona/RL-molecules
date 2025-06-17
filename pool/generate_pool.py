########## DDQN
# imports
import argparse
import csv
import os
import gymnasium as gym
import sys
import torch
from torch.utils.data import DataLoader
import wandb
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # Set RDKit's log level to error (eliminate warnings and info)
import pandas as pd
# Custom imports
import gym_rl_molecules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neural_network import NN_action_value
from ReplayBuffer_TDError import PrioritizedReplayBuffer, save_replay_buffer, load_replay_buffer
from training_functions_mp import custom_collate_fn, compute_avg_grad_norm, update_target_network_soft, update_target_network_hard, get_mask, apply_mask
from policy import policy_greedy, policy_multinomial
# Seeds
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

########################################################################################################

def arg_parser():
    parser = argparse.ArgumentParser()

    # Pool
    parser.add_argument('--num_unique', type=int, help='Number of unique molecules to be generated',
                        default=10000)
    # Results
    parser.add_argument('--use_wandb', type=bool, help='Whether to use wandb to track training',
                        default=False)
    parser.add_argument('--output_csv', type=str, help='Name of the output file to store the molecules',
                        default='pool')
    # Checkpoint
    parser.add_argument('--checkpoint_path', type=str, help='Absolute or relative path to the checkpoint',
                        default='example_checkpoint.pth')
    # Device
    parser.add_argument('--device', type=str, help='Device to run the model', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--num_cpus', type=int, help='Number of CPUs for the action masking computation', default=16)
    # Environment
    parser.add_argument('--fragments_file', type=str, help='Absolute or relative path to the set of fragments CSV',
                        default='../data/fragments.csv')
    parser.add_argument('--embedding_atoms', type=str, help='Atom types for the embedding process', default='C,N,O,F,S,Cl,Br')
    parser.add_argument('--addition_atoms', type=str, help='Atom types for scaffold addition', default='C,N,O,F')
    parser.add_argument('--max_episode_steps', type=int, help='Maximum number of steps per episode', default=16)
    parser.add_argument('--qed_target', type=float, help='Target QED value', choices=[0.7, 0.8, 0.9, 1], default=0.8)
    parser.add_argument('--initial_scaffold', type=str, help='Scaffold source', choices=['moses', 'carbon'], default='carbon')
    # Policy (DDQN)
    parser.add_argument('--size_emb', type=int, help='Embedding size', default=32)
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension size', default=32)
    parser.add_argument('--backward_actions', type=bool, help='Allow backward actions', default=True)
    parser.add_argument('--type_policy', type=str, help='Type of policy for action selection', choices=['greedy', 'multinomial'],
                        default='greedy')
    # Others
    parser.add_argument('--num_episodes', type=int, help='Total number of training episodes', default=50000)
    parser.add_argument('--render_molecules', type=bool, help='Whether to rend molecules during training', default=False)
    parser.add_argument('--render_mode', type=str, help='Type of molecule rendering', choices=['human', 'rgb_array'],
                        default='human')
    return parser.parse_args()

args = arg_parser()

########################################################################################################

# Set device
torch.set_default_device(args.device)

# Embedding: bond, atom and fragment types
fragments_pd = pd.read_csv(args.fragments_file, header=None)
embedding_fragmenttypes = list(fragments_pd[0])
embedding_bondtypes = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
embedding_atomtypes = args.embedding_atoms.split(',')

# Addition (molecule modification): atom and fragment types
addition_atomtypes = args.addition_atoms.split(',')
addition_fragmenttypes = list(fragments_pd[0])

# Reference dataset (MOSES)
moses_dataset = pd.read_csv("../data/MOSES_dataset.zip")
MOSES_train_split = list(moses_dataset[moses_dataset['SPLIT']=='train']['SMILES'])
reference_dataset = MOSES_train_split[:10000]

# List of initital scaffold(s) for molecule generation
if args.initial_scaffold == 'carbon':
    # De novo molecular design
    ls_scaffolds = ['C']
else:
    # Molecule optimization
    ls_scaffolds = MOSES_train_split[10000:20000]

# Results storage
# wandb
if args.use_wandb:
    wandb.init(
        project="DDQN training",
        config={
            "num_unique": args.num_unique,
            "checkpoint_path": args.checkpoint_path,
            "num_cpus": args.num_cpus,
            "max_episode_steps": args.max_episode_steps,
            "qed_target": args.qed_target,
            "size_emb": args.size_emb,
            "hidden_dim": args.hidden_dim,
            "backward_actions": args.backward_actions,
            "num_scaffolds": len(ls_scaffolds)
        },
    )

# 1) Environment
# Create the environment instance
env = gym.make('rl-molecules-v0')
# Mininum reward (for min-max normalization)
target2min = {0.7: 0.7407407407407407, 0.8: 0.7142857142857143, 0.9: 0.6896551724137931, 1:0.6666666666666666} # {qed_target1: min_reward1, qed_target2, min_reward2, ...}
min_reward = target2min[args.qed_target]
# Initialize the parameters of the environment (embedding and external)
env.initialize_parameters(embedding_atomtypes = embedding_atomtypes,
                          embedding_bondtypes = embedding_bondtypes,
                          embedding_fragmenttypes = embedding_fragmenttypes,
                          addition_atomtypes = addition_atomtypes,
                          addition_fragmenttypes = addition_fragmenttypes,
                          ls_scaffolds = ls_scaffolds,
                          max_episode_steps = args.max_episode_steps,
                          qed_target = args.qed_target,
                          min_reward = min_reward,
                          render_molecules = args.render_molecules,
                          render_mode=args.render_mode)

# 2) Neural networks (online and target)
net_kwargs = {
    'num_categories': len(embedding_atomtypes) + len(embedding_fragmenttypes),
    'size_emb': args.size_emb,
    'hidden_dim': args.hidden_dim,
    'types_edges': len(embedding_bondtypes),
    'num_external': len(addition_atomtypes) + len(addition_fragmenttypes),
    'backward_actions': args.backward_actions
}
online_network = NN_action_value(**net_kwargs).to(args.device)
target_network = NN_action_value(**net_kwargs).to(args.device)
# Load the checkpoint from file
if os.path.exists(args.checkpoint_path):
    checkpoint = torch.load(args.checkpoint_path)
else:
    raise FileNotFoundError(f"Checkpoint file not found at the provided path: {args.checkpoint_path}. Please provide a valid path.")
# Load the values
online_network.load_state_dict(checkpoint['online_network'])
target_network.load_state_dict(checkpoint['target_network'])
epsilon_greedy_episode = checkpoint['episode']
print(f"Checkpoint initialized")

# 3) Pool generation
# Initialization of variables to store the generated molecules
generated_smiles = []
qed_smiles = []
generated_smiles_unique = []
qed_smiles_unique = []
repeated = 0

while len(generated_smiles_unique) < args.num_unique:  # outer loop: iterate through episodes
    print(f"---- Molecule generation ----")
    # Reset the environment
    state, info = env.reset()
    state = state.to(args.device)  # Move initial state to GPU
    print(f"Initial scaffold: {info['smiles']}")

    # Initialize episode variables
    done = False
    step = 0
    episode_reward = 0
    episode_step_losses = []
    episode_grad_norms = []

    # Initial action mask
    mask = get_mask(info['mol_copy'], args.num_cpus, args.backward_actions) # In the first step (single atom 'C') the action masking is needed to avoid removing the C atom

    while not done:  # inner loop: iterates through steps until the episode is complete
        step += 1
        print(f"Step {step}")
        # Feed state to network to obtain Q-values
        loader = DataLoader([state, state], batch_size=1, collate_fn = custom_collate_fn)
        input_online = next(iter(loader))
        q_values = online_network(input_online)
        # Mask invalid actions
        q_values = apply_mask(mask, q_values)
        # choose an action with the policy
        if args.type_policy == "greedy":
            action, stop_prob = policy_greedy(logits=q_values, episode=epsilon_greedy_episode, 
                                              total_num_episodes=args.num_episodes, start=0.9, end=0.05, 
                                              num_external=net_kwargs['num_external'], num_nodes=info['Num_nodes'],
                                              backward_actions=args.backward_actions)
        else:
            action, stop_prob = policy_multinomial(logits=q_values, episode=epsilon_greedy_episode, 
                                              total_num_episodes=args.num_episodes, start=0.9, end=0.05, 
                                              num_external=net_kwargs['num_external'], num_nodes=info['Num_nodes'],
                                              backward_actions=args.backward_actions)

        print(f"    - Action: {action}")
        # take the action
        next_state, reward, terminated, truncated, info = env.step(action + (step,))
        done = terminated or truncated
        print(f"    - Acceptance: {info['action_accepted']} | QED: {info['QED']} | Reward: {reward} | Done: {done} | SMILES: {info['smiles']}")
        next_state = next_state.to(args.device)  # Move next state to GPU
        # Mask: create the mask of the next_state (mask invalid actions)
        mask = get_mask(info['mol_copy'], args.num_cpus, args.backward_actions)

        # update the state
        state = next_state
        episode_reward += reward
    
    if info['smiles'] not in generated_smiles_unique:
        generated_smiles_unique.append(info['smiles'])
        qed_smiles_unique.append(info['last_QED'])
        if args.use_wandb:
            wandb.log({
                "QED": info['last_QED']
            })
        is_repeated = False
    else:
        repeated+=1
        is_repeated = True
    generated_smiles.append(info['smiles'])
    qed_smiles.append(info['last_QED'])
    print(f"Episode ended | Repeated: {is_repeated} | Molecule SMILES: {info['smiles']}")

#### GENERATE OUTPUT CSV FILES ####
# CSV unique molecules
with open(f"{args.output_csv}_{epsilon_greedy_episode}_unique.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['SMILES', 'QED', 'Novel']) # Novelty = whether it was present in the training dataset or not
    for i in range(len(generated_smiles_unique)):
        novelty = not generated_smiles_unique[i] in reference_dataset
        writer.writerow([generated_smiles_unique[i], qed_smiles_unique[i], novelty])

# CSV all molecules
with open(f"{args.output_csv}_{epsilon_greedy_episode}_all.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([f"Repeated: {repeated}"])
    writer.writerow(['SMILES', 'QED', 'Novel']) # Novelty = whether it was present in the training dataset or not
    for i in range(len(generated_smiles)):
        novelty = not generated_smiles[i] in reference_dataset
        writer.writerow([generated_smiles[i], qed_smiles[i], novelty])

if args.use_wandb:
    wandb.finish()
print("Training finished")
