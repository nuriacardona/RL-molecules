# IMPORTS
import numpy as np
import random
import torch

#seeds
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


##### Policies

# Epsilon-greedy
def policy_greedy(logits, episode, total_num_episodes, start, end, num_external, num_nodes, backward_actions):
    """Epsilon-greedy policy to select an action from the Q-values produced by an action-value function

    Args:
        logits (pytorch tensor): raw q-values from the forward pass of the neural network.
        episode (int): current episode number.
        total_num_episodes (int): total number of episodes expected in training.
        start (float): initial epsilon value.
        end (float): final epsilon value.
        num_external (int): number of atoms and fragments (scaffolds) that can be added to the molecule.
        num_nodes (int): current number of nodes in the molecule.
        backward_actions (boolean): if the model is allowed to revert actions or not.

    Returns:
        action (tuple): action type, action details and action index.
        float: stop probability (logits_softmax[0]).
    """

    # Apply softmax to the input logits
    logits_softmax = torch.softmax(logits, dim=0)

    # Random sample between 0 and 1
    sample = random.random()
    # Threshold value epsilon for this episode
    if episode < total_num_episodes:
        # Linear annealing of epsilon
        epsilon = np.linspace(start, end, total_num_episodes)[episode]
    else: # if starting from an existing checkpoint, the episode may be higher than the original total
        epsilon = end
    
    # Select the action (determine the action index)
    if sample < epsilon:
        # Boolean to ensure the selected action is valid (i.e. not masked to 0)
        not_valid = True 
        while not_valid:
            # Random action (with probability epsilon a random action is taken)
            action_idx = random.choice(range(len(logits_softmax)))
            # If the logit is bigger than 0 it means the action was valid (not masked to 0)
            if logits_softmax[action_idx] > 0:
                not_valid = False
    else:
        # Highest Q-value action is taken (argmax) (with probability 1-epsilon)
        action_idx = torch.argmax(logits_softmax).item()

    # Define the total number of possible actions that can be potentially applied to each node (number of logits that correspond to actions involving each node)
    if backward_actions:
        # If backward actions are allowed, it is necessary to add 1 to the number of external (in order to take into account the 'backward' action for each node)
        total_actions_per_node = num_external+1
    else:
        # Each node can only have addition actions (no backward actions, i.e. the node can't be removed)
        total_actions_per_node = num_external

    # Retrieve the action details
    if action_idx == 0:
        action_type = "Stop"
        action_details = None
    elif action_idx <= (total_actions_per_node)*num_nodes: 
        # Determine the node (subtract 1 to 'action_idx' to account for the Stop action)
        node_idx = (action_idx - 1) // (total_actions_per_node) 
        # Determine the node category
        external_idx = (action_idx - 1) % (total_actions_per_node)
        # Determine the action type and its details
        if external_idx == num_external:
            action_type = "RemoveNode"
            action_details = {"node_idx": int(node_idx)}
        else:
            action_type = "AddNode"
            action_details = {"node_idx": int(node_idx), "external_idx": int(external_idx)}
    else:
        action_type = "AddEdge"
        # Adjust index for Stop and AddNode logits
        nonedge_idx = action_idx - 1 - (total_actions_per_node)*num_nodes
        # Index in the non-edge list
        action_details = {"nonedge_idx": nonedge_idx}

    # Build the action tuple
    action = (action_type, action_details, int(action_idx))

    return action, logits_softmax[0]


# Multinomial
def policy_multinomial(logits, episode, total_num_episodes, start, end, num_external, num_nodes, backward_actions):
    """Multinomial policy to select an action from the Q-values produced by an action-value function

    Args:
        logits (pytorch tensor): raw q-values from the forward pass of the neural network.
        episode (int): current episode number.
        total_num_episodes (int): total number of episodes expected in training.
        start (float): initial epsilon value.
        end (float): final epsilon value.
        num_external (int): number of atoms and fragments (scaffolds) that can be added to the molecule.
        num_nodes (int): current number of nodes in the molecule.
        backward_actions (boolean): if the model is allowed to revert actions or not.

    Returns:
        action (tuple): action type, action details and action index.
        float: stop probability (logits_softmax[0]).
    """

    # Apply softmax to the input logits
    logits_softmax = torch.softmax(logits, dim=0)
    # Create a multinomial distribution with the logits and sample the action index
    action_idx = torch.multinomial(logits_softmax, 1)
    # Convert the generated tensor to an integer value
    action_idx = action_idx.item()

    # Define the total number of possible actions that can be potentially applied to each node (number of logits that correspond to actions involving each node)
    if backward_actions:
        # If backward actions are allowed, it is necessary to add 1 to the number of external (in order to take into account the 'backward' action for each node)
        total_actions_per_node = num_external+1
    else:
        # Each node can only have addition actions (no backward actions, i.e. the node can't be removed)
        total_actions_per_node = num_external

    # Retrieve the action details
    if action_idx == 0:
        action_type = "Stop"
        action_details = None
    elif action_idx <= (total_actions_per_node)*num_nodes: 
        # Determine the node (subtract 1 to 'action_idx' to account for the Stop action)
        node_idx = (action_idx - 1) // (total_actions_per_node) 
        # Determine the node category
        external_idx = (action_idx - 1) % (total_actions_per_node)
        # Determine the action type and its details
        if external_idx == num_external:
            action_type = "RemoveNode"
            action_details = {"node_idx": int(node_idx)}
        else:
            action_type = "AddNode"
            action_details = {"node_idx": int(node_idx), "external_idx": int(external_idx)}
    else:
        action_type = "AddEdge"
        # Adjust index for Stop and AddNode logits
        nonedge_idx = action_idx - 1 - (total_actions_per_node)*num_nodes
        # Index in the non-edge list
        action_details = {"nonedge_idx": nonedge_idx}

    # Build the action tuple
    action = (action_type, action_details, int(action_idx))

    return action, logits_softmax[0]