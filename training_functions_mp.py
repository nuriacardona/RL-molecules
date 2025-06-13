# IMPORTS
import copy
import multiprocessing as mp
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import rdmolops
import torch
from torch_geometric.data import Batch

#seeds
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

#### Batch formation        
def custom_collate_fn(data_ls):
    """Custom collate function to handle 'non_edges' attribute and create 'batch_non_edge'.

    Args:
        data_ls (list of Data instances): List of Data objects to batch.

    Returns:
        Batch Data object with standard attributes, 'non_edges' and 'batch_non_edge'.
    """

    # Generate standard batch attributes (e.g. 'x' and 'edge_index')
    batch = Batch.from_data_list(data_ls)
    
    # Define lists to store non_edges and their corresponding graph indices
    nonedges_ls = []
    batch_nonedge_ls = []
    # Cumulative node count to adjust indices
    cumsum_nodes = 0
    # Start at graph index 0
    graph_idx = 0
    # Iterate over the data objects contain in the input list
    for data in data_ls:
        # See if the current data object has the 'non_edges' attribute and it is not None
        if hasattr(data, 'non_edges') and data.non_edges is not None:
            # If non_edges is not empty (number of elements higher than 0)
            if data.non_edges.numel() > 0:
                # Correct the indices of the non_edges by the cumulative nodes
                nonedges_idx = data.non_edges + cumsum_nodes
                # Add to the non-edges list
                nonedges_ls.append(nonedges_idx)
                # Tensor to store the graph idx corresponding to each non-edge
                graph_idxs = torch.full((data.non_edges.size(0),), graph_idx, dtype=torch.long)
                # Add to the non-edges batch list
                batch_nonedge_ls.append(graph_idxs)
        # Update cumulative node count and graph idx
        cumsum_nodes += data.num_nodes
        graph_idx += 1
    
    if nonedges_ls:
        # Combine all the corrected non_edges tensors into a single one
        batch.non_edges = torch.cat(nonedges_ls, dim=0)
        # Combine all graph idx tensors into one
        batch.batch_non_edge = torch.cat(batch_nonedge_ls, dim=0)
    else:
        # If there are not non_edges in the entire batch then create empty tensors
        batch.non_edges = torch.empty((0, 2), dtype=torch.long)
        batch.batch_non_edge = torch.empty((0,), dtype=torch.long)
    
    return batch




#### Average gradient norm
def compute_avg_grad_norm(model):
    """Compute the average L2 norm of the gradients
    
    Args:
        model (NN_action_value): pytorch model.
    
    Returns:
        int indicating the average gradient norm (L2)
    """

    # Initialize a variable to store the total norm
    total_norm = 0.0
    # Count of the number of parameters visited
    num_params = 0
    # Iterate over the model parameters
    for param in model.parameters():
        # Verify that the gradient of the parameter is not None
        if param.grad is not None:
            # Add the gradient of the current parameter to the total
            total_norm += param.grad.norm(2).item()
            # Increase the count of analyzed parameters
            num_params +=1
    # Compute and return the avg norm
    return total_norm / num_params if num_params > 0 else 0.0




#### Target network updates
@torch.no_grad()
def update_target_network_soft(online_network, target_network, tau=0.005):
    """Perform a soft update of the target network
    
    Args:
        online_network (NN_action_value): online network with the updated parameters by gradient descent
        target_network (NN_action_value): target network to be updated
    """

    # Obtain the state dictionary of the online network
    online_statedict = online_network.state_dict()
    # Obtain the state dictionary of the target network
    target_statedict = target_network.state_dict()

    # Iterate over the model parameters
    for key in online_statedict:
        # Calculate the updated state dict into the target network
        target_statedict[key] = online_statedict[key]*tau + target_statedict[key]*(1-tau)
    # Load the updated state dict into the target network
    target_network.load_state_dict(target_statedict)

    return None

@torch.no_grad()
def update_target_network_hard(online_network, target_network, absolute_step, target_update_frequency):
    """Perform a hard update of the target network
    
    Args:
        online_network (NN_action_value): online network with the updated parameters by gradient descent
        target_network (NN_action_value): target network to be updated
        absolute_step (int): cumulative sum of all steps performed since the beginning of the training process
        target_update_frequency (int): frequency (number of episodes) with which the target network is hard updated
    """

    # See it is necessary to update the target in the current step
    if absolute_step % target_update_frequency == 0:
        # Hard-copy the parameters of the online network into the target network
        target_network.load_state_dict(online_network.state_dict())



#### Action masking (multiprocessed)
def process_addnode_task(inp):
    """Evaluate if an "Add node" action is valid or not

    Args:
        inp (tuple): details of the action to be checked
    
    Returns:
        tuple with the mask index and validity (1 is valid, 0 invalid)
    """

    source_idx, external_idx, Mol, backward_actions = inp
    # Retrieve the external atom/fragment to be added
    external = Mol.ls_external[external_idx]
    # Make a copy of the input molecule (otherwise the original is modified)
    Mol_copy = copy.deepcopy(Mol)

    # Try performing the action
    if external in Mol.addition_atomtypes: # 'external' is an atom
        # Add the atom
        nx_idx_2 = Mol_copy.add_atom(external)
        # Not necessary to remove an index
        rdkit_idx_remove = None
    else: # 'external' is a fragment
        # Add the fragment
        nx_idx_2, rdkit_idx_remove = Mol_copy.add_fragment(external)

    # Add bond
    new_bond = Mol_copy.add_bond(source_idx, nx_idx_2, rdkit_idx_remove, Chem.BondType.SINGLE)

    # Determine the action index (i.e. position of the action being evaluated in the mask)
    if backward_actions:
        # Add 1 to the number of external to account for the backward action
        action_idx = (source_idx*(Mol_copy.get_numexternal()+1))+external_idx+1
    else:
        # No need to add 1 to the number of external
        action_idx = (source_idx*Mol_copy.get_numexternal())+external_idx+1
    
    # Check if the performed action (atom or fragment addition) is valid
    if Mol_copy.check_molecule():
        return (action_idx, 1)
    else:
        return (action_idx, 0)

def process_addedge_task(inp):
    """ Evaluate if an "Add edge" action is valid or not

    Args:
        inp (tuple): details of the action to be checked
    
    Returns:
        tuple with the mask index and validity (1 is valid, 0 invalid)
    """

    # Unpack the details of the input action
    nonedge_idx, (nx_idx_1, nx_idx_2), Mol, backward_actions = inp
    # Retrieve the index of the first node involved in the edge
    nx_idx_1 = nx_idx_1.item()
    # Retrieve the index of the second node involved in the edge
    nx_idx_2 = nx_idx_2.item()
    # Make a copy of the input molecule (otherwise the original is modified)
    Mol_copy = copy.deepcopy(Mol)

    # Add the bond (the current non-edge)
    new_bond = Mol_copy.add_bond(nx_idx_1, nx_idx_2, None, Chem.BondType.SINGLE)

    # Determine the action index (i.e. position of the action being evaluated in the mask)
    if backward_actions:
        # Add 1 to the number of external to account for the backward action
        action_idx = (Mol_copy.get_numnodes()*(Mol_copy.get_numexternal()+1))+nonedge_idx+1
    else:
        # No need to add 1 to the number of external
        action_idx = (Mol_copy.get_numnodes()*Mol_copy.get_numexternal())+nonedge_idx+1
    
    # Check if the performed action (add bond) is valid
    if Mol_copy.check_molecule():
        return (action_idx, 1)
    else:
        return (action_idx, 0)
    
def process_remove_task(inp):
    """Evaluate if a "Remove node" action is valid or not

    Args:
        inp (tuple): details of the action to be checked
    
    Returns:
        tuple with the mask index and validity (1 is valid, 0 invalid)
    """

    # Unpack the details of the input action
    nx_idx, Mol = inp
    # Make a copy of the input molecule (otherwise the original is modified)
    Mol_copy = copy.deepcopy(Mol)

    # Remove the component (nx_idx)
    Mol_copy.remove_component(nx_idx)
    # Check how many fragments are
    Mol_copy_fragments = rdmolops.GetMolFrags(Mol_copy.rdkit)
    # Determine the action index (i.e. position of the action being evaluated in the mask)
    action_idx = (nx_idx+1)*(Mol_copy.get_numexternal()+1)

    # If there was no fragmentation (i.e. the molecule is still a single entity)
    if len(Mol_copy_fragments) == 1:
        # Check if the resulting molecule (without nx_idx) is valid
        if Mol_copy.check_molecule():
            # First element of the tuple is the index in the masking; second element is validity (mark as valid)
            return (action_idx, 1)
    # First element of the tuple is the index in the masking; second element is validity (mark as invalid)
    return (action_idx, 0) # Mark as invalid

def get_mask(Mol, num_cpus, backward_actions):
    """Generate mask for q-values (1 means valid and 0 means invalid).

    Args:
        Mol (Molecule): molecule to test the actions.
        num_cpus (int): number of available CPUs for parallelization.
        backward_actions (boolean): if backward actions (remove a node) are allowed.
    
    Returns:
        mask (tensor): whether each actions is valid or not.
    """

    ## Initialize the mask (full of 0s) and the molecule variables
    num_nodes = Mol.get_numnodes()
    num_external = Mol.get_numexternal()
    num_nonedges = Mol.get_nonedges().size(0)
    # Mask size (depending if backward actions are allowed or not)
    if backward_actions:
        # account for backward action
        mask = torch.zeros(1+(num_nodes*(num_external+1))+num_nonedges)
    else:
        # do not account for  backward action
        mask = torch.zeros(1+(num_nodes*num_external)+num_nonedges)

    ## Stop logit: terminal state is always valid
    mask[0] = 1

    ## Addnode and remove logits
    # Define the Addnode tasks
    addnode_tasks = [(source_idx, external_idx, Mol, backward_actions)
                        for source_idx in range(num_nodes)
                        for external_idx in range(Mol.get_numexternal())]
    # Create a pool of processes
    with mp.Pool(num_cpus) as pool:
        results = pool.map(process_addnode_task, addnode_tasks)
    # Fill the addnode actions
    for action_idx, is_valid in results:
        mask[action_idx] = is_valid
    # Define the remove tasks
    if backward_actions:
        # To remove the original C is not a valid action
        if num_nodes > 1:
            remove_tasks = [(nx_idx, Mol) for nx_idx in range(1, num_nodes)]
            # Generate a pool of processes
            with mp.Pool(num_cpus) as pool:
                remove_results = pool.map(process_remove_task, remove_tasks)
            # Fill the remove node actions
            for removeaction_idx, removeis_valid in remove_results:
                mask[removeaction_idx] = removeis_valid

    ## Addedge logits
    # Check if there are addedge logits or not
    if num_nonedges > 0:
        # Define the addedge tasks (i.e. the addedge actions to be validated)
        edge_tasks = [(nonedge_idx, (nx_idx_1, nx_idx_2), Mol, backward_actions) 
                        for nonedge_idx, (nx_idx_1, nx_idx_2) in enumerate(Mol.get_nonedges())]
        # Create a pool of processes
        with mp.Pool(num_cpus) as pool:
            results = pool.map(process_addedge_task, edge_tasks)
        # Fill the addedge actions
        for action_idx, is_valid in results:
            mask[action_idx] = is_valid
    
    # Return the final mask
    return mask

def apply_mask(mask, q_values):
    """Apply a mask to a set of Q-values

    Args:
        mask (tensor): tensor of the same size as the Q-values indicating valid and invalid actions.
        q_values (tensor): q-values to apply the mask.
    
    Returns:
        tensor: masked q-values
    """

    # Invalid actions are set to -1e9
    return q_values.squeeze() + (mask-1)*1e9