########## DDQN
# imports
import argparse
import os
import warnings
import gymnasium as gym
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # Set RDKit's log level to error (eliminate warnings and info)
from collections import deque
from statistics import mean
import pandas as pd
# Custom imports
import gym_rl_molecules
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

    # Results
    parser.add_argument('--checkpoint_dir', type=str, help='Directory to store the model checkpoints',
                        default='checkpoints')
    parser.add_argument('--use_wandb', type=bool, help='Whether to use wandb to track training',
                        default=False)
    parser.add_argument('--frequency_checkpoint', type=int, help='Periodic checkpoint save frequency (episodes)',
                        default=500)
    parser.add_argument('--episodes_before_checkpoints', type=int, help='Episodes before collecting checkpoints',
                        default=10000)
    parser.add_argument('--qed_window', type=int, help='Window of most recent episodes to evaluate QED improvement',
                        default=300)
    parser.add_argument('--reward_window', type=int, help='Window of most recent episodes to evaluate reward improvement',
                        default=300)
    # Checkpoint
    parser.add_argument('--start_from_checkpoint', type=bool, help='Whether to start from a previous checkpoint',
                        default=False)
    parser.add_argument('--checkpoint_path', type=str, help='Absolute or relative path to the checkpoint',
                        default='')
    parser.add_argument('--checkpoint_rb_path', type=str, help='Absolute or relative path to the replay buffer checkpoint',
                        default='')
    # Device
    parser.add_argument('--device', type=str, help='Device to run the model; "cpu" or "cuda"', default='cpu')
    parser.add_argument('--num_cpus', type=int, help='Number of CPUs for the action masking computation', default=16)
    # Environment
    parser.add_argument('--fragments_file', type=str, help='Absolute or relative path to the set of fragments CSV',
                        default='fragments.csv')
    parser.add_argument('--embedding_atoms', type=str, help='Atom types for the embedding process', default='C,N,O,F,S,Cl,Br')
    parser.add_argument('--addition_atoms', type=str, help='Atom types for scaffold addition', default='C,N,O,F')
    parser.add_argument('--max_episode_steps', type=int, help='Maximum number of steps per episode', default=16)
    parser.add_argument('--qed_target', type=float, help='Target QED value', choices=[0.7, 0.8, 0.9, 1], default=0.8)
    parser.add_argument('--initial_scaffold', type=str, help='Scaffold source', choices=['moses', 'carbon'], default='carbon')
    # Policy (DDQN)
    parser.add_argument('--size_emb', type=int, help='Embedding size', default=32)
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension size', default=32)
    parser.add_argument('--backward_actions', type=bool, help='Allow backward actions', default=True)
    parser.add_argument('--target_updates', type=str, choices=['hard', 'soft'], help='Type of target network updates',
                        default='hard')
    parser.add_argument('--target_hard_update_frequency', type=int, help='Frequency of hard target network updates',
                        default=10000)
    parser.add_argument('--type_policy', type=str, help='Type of policy for action selection', choices=['greedy', 'multinomial'],
                        default='greedy')
    # Training
    parser.add_argument('--lr_optimizer', type=float, help='Learning rate (optimizer)', default=0.00025)
    parser.add_argument('--num_episodes', type=int, help='Total number of training episodes', default=50000)
    parser.add_argument('--max_norm', type=float, help='Gradient clipping threshold', default=1.0)
    parser.add_argument('--gamma', type=float, help='Target Q-value (factor)', default=0.99)
    parser.add_argument('--rb_capacity', type=int, help='Replay buffer total capacity', default=10000)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=64)
    parser.add_argument('--synthetic_experiences', type=bool, help='Whether to load synthetic experiences to the RB',
                        default=False)
    parser.add_argument('--synthetic_experiences_path', type=str, help='Absolute or relative path to synthetic experiences',
                        default='')
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

# List of initital scaffold(s) for molecule generation
if args.initial_scaffold == 'carbon':
    # De novo molecular design
    ls_scaffolds = ['C']
else:
    moses_dataset = pd.read_csv("MOSES_dataset.csv")
    # Molecule optimization
    ls_scaffolds = list(moses_dataset[moses_dataset['SPLIT'] == 'train'][10000:20000]['SMILES'])


# Results storage
# wandb
if args.use_wandb:
    wandb.init(
        project="DDQN training",
        config={
            "checkpoint_dir": args.checkpoint_dir,
            "frequency_checkpoint": args.frequency_checkpoint,
            "start_from_checkpoint": args.start_from_checkpoint,
            "num_cpus": args.num_cpus,
            "max_episode_steps": args.max_episode_steps,
            "qed_target": args.qed_target,
            "size_emb": args.size_emb,
            "hidden_dim": args.hidden_dim,
            "target_updates": args.target_updates,
            "target_hard_update_frequency": args.target_hard_update_frequency,
            "lr_optimizer": args.lr_optimizer,
            "max_norm": args.max_norm,
            "backward_actions": args.backward_actions,
            "num_scaffolds": len(ls_scaffolds),
            "num_episodes": args.num_episodes,
            "rb_capacity": args.rb_capacity,
            "batch_size": args.batch_size,
            "synthetic_experiences": args.synthetic_experiences,
        },
    )
# Checkpoints: create the folder to store the checkpoints if it doesn't exist
os.makedirs(args.checkpoint_dir, exist_ok=True)

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

# 2) Neural network, optimizer and replay buffer
# Neural network parameters (online and target)
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
absolute_step = 0
optimizer = optim.AdamW(online_network.parameters(), lr=args.lr_optimizer, weight_decay=0.001)
if args.target_updates not in ['hard', 'soft']:
    raise ValueError(f"Invalid value for 'target_updates': {args.target_updates}.\nVariable 'target_updates' must be either 'hard' or 'soft'.")

# CHECKPOINT AND PREVIOUS KNOWLEDGE
# Warning in case there is a flag conflict
if args.start_from_checkpoint and args.synthetic_experiences:
    warnings.warn(
        "Flag conflict: 'start_from_checkpoint=True' and 'synthetic_experiences=True' are mutually exclusive. "
        "'synthetic_experiences' will be ignored (i.e. no synthetic experiences will be loaded, while the checkpoint will be loaded)."
    )
if args.start_from_checkpoint:
    #### Checkpoint
    # Load the checkpoint from file
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
    else:
        raise FileNotFoundError(f"Checkpoint file not found at the provided path: {args.checkpoint_path}. Please provide a valid path.")
    # Load the replay buffer from file
    if os.path.exists(args.checkpoint_rb_path):
        replay_buffer = load_replay_buffer(filename=args.checkpoint_rb_path)
    else:
        raise FileNotFoundError(f"Replay Buffer file not found at the provided path: {args.checkpoint_rb_path}. Please provide a valid path.")
    # Load the values
    online_network.load_state_dict(checkpoint['online_network'])
    target_network.load_state_dict(checkpoint['target_network'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    starting_episode = checkpoint['episode']
    print(f"Checkpoint initialized")
else:
    #### New execution (without relying on previous checkpoints)
    starting_episode = 0
    # Initialize the target network to the same values as the online network
    target_network.load_state_dict(online_network.state_dict())
    replay_buffer = PrioritizedReplayBuffer(size_buffer = args.rb_capacity, epsilon = 0.01, alpha=0.6, beta=0.4, increase=0.001)
    # If synthetic experiences (RB 'memory') need to be loaded
    if args.synthetic_experiences:
        # Check if the provided path exists
        if os.path.exists(args.synthetic_experiences_path):
            # Load the list of synthetic experiences
            synth_exp = torch.load(args.synthetic_experiences_path)
            # Store the synthetic experiences in the replay buffer memory
            replay_buffer.memory = deque(synth_exp[:5000], maxlen=replay_buffer.capacity)
            # Initialize the priorities (the priority of all experiences is set to 1)
            replay_buffer.priorities = deque([1]*len(replay_buffer.memory), maxlen=replay_buffer.capacity)
            print(f"Synthetic experiences were loaded | RB memory size: {len(replay_buffer.memory)}")
        else:
            raise FileNotFoundError(f"Synthetic experiences file not found at the provided path: {args.synthetic_experiences_path}.\nPlease provide a valid path.")  
    print(f"No checkpoint was loaded")  
print(f"Training will start from episode {starting_episode}")

# 4) Training
# QED and mean reward historical (last 1000 episodes) -> to decide if a checkpoint is saved
QED_queue = deque(maxlen=args.qed_window)
reward_queue = deque(maxlen=args.reward_window)

# Training loop
for episode in range(starting_episode, starting_episode+args.num_episodes):  # outer loop: iterate through episodes
    print(f"\n-------- EPISODE {episode} --------")

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
    replay_buffer.update_beta()

    # Initial action mask
    mask = get_mask(info['mol_copy'], args.num_cpus, args.backward_actions) # In the first step (single atom 'C') the action masking is needed to avoid removing the C atom

    while not done:  # inner loop: iterates through steps until the episode is complete
        optimizer.zero_grad()
        step += 1
        absolute_step += 1
        print(f"Step {step}")
        # Feed state to network to obtain Q-values
        loader = DataLoader([state, state], batch_size=1, collate_fn = custom_collate_fn)
        input_online = next(iter(loader))
        q_values = online_network(input_online)
        # Mask invalid actions
        q_values = apply_mask(mask, q_values)
        # choose an action with the policy
        if args.type_policy == "greedy":
            action, stop_prob = policy_greedy(logits=q_values, episode=episode, 
                                              total_num_episodes=args.num_episodes, start=0.9, end=0.05, 
                                              num_external=net_kwargs['num_external'], num_nodes=info['Num_nodes'],
                                              backward_actions=args.backward_actions)
        else:
            action, stop_prob = policy_multinomial(logits=q_values, episode=episode, 
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
        # Store the last experience in the replay buffer
        replay_buffer.store_experience(state, action, reward, next_state, done, mask)

        # Training begins once the buffer exceeds the desired batch size
        if replay_buffer._get_size() >= args.batch_size:
            print("    - Training")
            #### REPLAY BUFFER
            # Take a sample
            states, actions, rewards, next_states, dones, masks, rb_indices, weights = replay_buffer.sample_batch_experiences(args.batch_size)
            # Move data to device
            states = [s.to(args.device) for s in states]
            next_states = [ns.to(args.device) for ns in next_states]
            rewards = rewards.to(args.device)
            dones = dones.to(args.device)
            masks = [mask.to(args.device) for mask in masks]

            #### STATES: current state Q-value (Q-value indexed by the selected action)
            ## Data Loader
            states_loader = DataLoader(list(states), batch_size=len(states), collate_fn = custom_collate_fn)
            states_batch = next(iter(states_loader))  # there will only be 1 batch
            ## Forward pass (get the q-values)
            states_qvalues = online_network(states_batch)
            ## Index the q-values of interest
            # Row indices: range until batch_size-1
            row_indices = torch.arange(args.batch_size)
            # Column indices: correspond to the action indices (last element from each action tuple)
            column_indices = torch.tensor(list(map(lambda x: x[-1], actions)))
            # Retrieve the elements
            current_q_values = states_qvalues[row_indices, column_indices]

            #### NEXT STATES AND REWARD: TD-target
            with torch.no_grad():
                ## Data Loader
                nextstates_loader = DataLoader(list(next_states), batch_size=len(next_states), collate_fn = custom_collate_fn)
                nextstates_batch = next(iter(nextstates_loader))
                ## Forward pass (online network) -> determine the actions (indices)
                nextstates_qvalues_online = online_network(nextstates_batch)
                # Apply padding so that all masks have the same dimensions (size)
                masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
                ## Apply masking to the logits
                nextstates_qvalues_online = apply_mask(masks, nextstates_qvalues_online)
                # Column indices: argmax of each row
                column_indices = torch.argmax(nextstates_qvalues_online, dim=1)
                ## Forward pass (target network) -> determine the value of these actions (q-value)
                nextstates_qvalues_target = target_network(nextstates_batch)
                ## Target Q-values
                target_q_values = rewards + args.gamma * nextstates_qvalues_target[row_indices, column_indices] * (1-dones.float())

            # Update the Replay Buffer priorities for that batch
            td_errors = target_q_values - current_q_values
            replay_buffer.update_priorities(rb_indices, td_errors)

            #### LOSS COMPUTATION
            loss = torch.mean(weights * (td_errors) ** 2)
            # loss = F.mse_loss(current_q_values, target_q_values)
            # Backpropagation (compute the gradients)
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(online_network.parameters(), args.max_norm)

            # Compute the L2 norm of the clipped gradients
            avg_grad_norm = compute_avg_grad_norm(online_network)
            episode_grad_norms.append(avg_grad_norm)

            # Update the parameters with the clipped gradients
            optimizer.step()
            # Update the target network weights
            if args.target_updates == 'hard':
                update_target_network_hard(online_network, target_network, absolute_step, args.target_hard_update_frequency)
            elif args.target_updates == 'soft':
                update_target_network_soft(online_network, target_network)
            episode_step_losses.append(loss.item())
            print(f"        · Loss: {loss.item()}")
        # update the state
        state = next_state
        episode_reward += reward
    # WANDB: Log metrics to wandb
    if args.use_wandb:
        avg_loss = mean(episode_step_losses) if len(episode_step_losses) > 0 else 0.0
        print(f"Average episode loss: {avg_loss}")
        avg_grad_norm_episode = mean(episode_grad_norms) if len(episode_grad_norms) > 0 else 0.0
        wandb.log({
            "Episode": episode,
            "Reward": episode_reward,
            "Mean Episode Reward": mean(info["StepRewards"]),
            "Loss": avg_loss,
            "Num steps": step,
            "Avg Grad Norm (L2)": avg_grad_norm_episode,
            "Final TPSA": info["last_TPSA"],
            "Final MW": info["last_MW"],
            "Final LogP": info["last_LogP"],
            "Final QED": info["last_QED"],
            "Final number of atoms of the episode": info["Num_atoms"],
            "Stop probability": stop_prob
        })
    # CHECKPOINT: Decide if a checkpoint is saved
    save_checkpoint = False
    if episode > starting_episode+args.episodes_before_checkpoints:
        improved_qed = info["last_QED"] > max(QED_queue)
        improved_reward = mean(info["StepRewards"]) > max(reward_queue)
        periodic_checkpoint = episode % args.frequency_checkpoint == 0
        print(f"Improved QED: {improved_qed} | Improved reward: {improved_reward} | Periodic checkpoint: {periodic_checkpoint}")
        if improved_qed or improved_reward or periodic_checkpoint:
            save_checkpoint = True
    if save_checkpoint:
        torch.save({
            'episode': episode,
            'online_network': online_network.state_dict(),
            'target_network': target_network.state_dict(),
            'optimizer': optimizer.state_dict()
            },
            os.path.join(args.checkpoint_dir, f'checkpoint_{episode}.pth'))
        save_replay_buffer(replay_buffer, os.path.join(args.checkpoint_dir, f'replaybuffer_{episode}.pth'))
    # Update the QED and reward queues
    QED_queue.append(info["last_QED"]) # add the QED value to the queue
    reward_queue.append(mean(info["StepRewards"])) # add the mean reward to the queue
if args.use_wandb:
    wandb.finish()
print("Training finished")
