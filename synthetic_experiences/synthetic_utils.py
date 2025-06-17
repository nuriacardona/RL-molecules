# Imports
import pandas as pd
import csv
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Descriptors import qed, MolLogP

from rdkit.Chem.BRICS import BRICSDecompose, BRICSBuild
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # Disable RDKit warnings
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import random
from statistics import mean
import re
import sys
import os
from molecule_class import Molecule
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training_functions_mp import get_mask
import copy
import torch

def remove_dummy_numbers(s):
    """
    Remove the number of the dummy atoms.

    Args:
        s (str): Input string.
    
    Returns:
        str: String with numbers before '*' removed. Example: from [3*] to [*]
    """
    return re.sub(r'\[(\d+)\*', '[*', s)

def perform_action(Mol, input_action):
    max_episode_steps = 16
    #### Action (modification) to be performed in the molecule
    action_type, action_details, action_idx = input_action

    # ADDNODE ACTION: EXTERNAL ATOM/FRAGMENT ADDITION
    if action_type == "AddNode":
        # NODE 1 INDEX IDENTIFICATION: Existing node in the molecule
        nx_idx_1 = action_details["node_idx"] # nxgraph index

        # NODE 2 INDEX IDENTIFICATION: External atom/fragment to be added to the molecule
        external_idx = action_details["external_idx"] # nxgraph index
        # a) Add Atom
        if external_idx < len(Mol.addition_atomtypes):
            atom = Mol.addition_atomtypes[external_idx]
            nx_idx_2 = Mol.add_atom(atom)
            rdkit_idx_remove = None
        # b) Add Fragment
        else:
            fragment = Mol.addition_fragmenttypes[external_idx-len(Mol.addition_atomtypes)]
            # fragment = self.ls_fragmenttypes[external_idx-len(self.ls_atomtypes)+1]
            nx_idx_2, rdkit_idx_remove = Mol.add_fragment(fragment)
    
    # REMOVENODE ACTION: ELIMINATE AN EXISTING ATOM/FRAGMENT
    elif action_type == "RemoveNode":
        removed_nx_idx = action_details["node_idx"]
        Mol.remove_component(removed_nx_idx)
    
    # ADDEDGE ACTION: NON-EDGE IDENTIFICATION
    elif action_type == "AddEdge":
        nonedge_idx = action_details["nonedge_idx"]
        nonedges_nxgraph = Mol.get_nonedges()
        target_nxnonedge = nonedges_nxgraph[nonedge_idx]
        nx_idx_1 = target_nxnonedge[0].item()
        nx_idx_2 = target_nxnonedge[1].item()
        rdkit_idx_remove = None

    # ADDNODE AND ADDEDGE ACTIONS: BOND CREATION
    if (action_type == "AddNode") or (action_type == "AddEdge"):
        bond_type = Chem.BondType.SINGLE
        new_bond = Mol.add_bond(nx_idx_1, nx_idx_2, rdkit_idx_remove, bond_type)

    # STOP ACTION: SET EPISODE TERMINATION TO TRUE
    if (action_type == "Stop"):
        episode_term = True
    else:
        episode_term = False

    return new_bond, episode_term

def get_reward(Mol, qed_target, min_reward):
    # Compute QED
    try:
        qed_mol = qed(Mol.rdkit)
    except:
        qed_mol = 0
    # Reward
    alpha = 0.5
    reward = 1/(1+alpha*abs(qed_mol-qed_target))
    # Min-max normalization [0,1]
    max_reward = 1.0
    scaled_reward = (reward - min_reward) / (max_reward - min_reward)
    return scaled_reward

def check_action_idx(action_idx, total_actions_per_node, num_nodes, num_external):
    # Retrieve the action details
    if action_idx == 0:
        action_type = "Stop"
        action_details = None
    elif action_idx <= (total_actions_per_node)*num_nodes: 
        node_idx = (action_idx - 1) // (total_actions_per_node)  # Determine the node (subtract 1 to 'action_idx' for Stop action)
        external_idx = (action_idx - 1) % (total_actions_per_node)  # Determine the node category
        if external_idx == num_external:
            action_type = "RemoveNode"
            action_details = {"node_idx": int(node_idx)}
        else:
            action_type = "AddNode"
            action_details = {"node_idx": int(node_idx), "external_idx": int(external_idx)}
    else:
        action_type = "AddEdge"
        nonedge_idx = action_idx - 1 - (total_actions_per_node)*num_nodes  # Adjust index for Stop and AddNode logits
        action_details = {"nonedge_idx": nonedge_idx}  # Index in the non-edge list
    
    action = (action_type, action_details, action_idx)

    return action


# Saving synthetic experiences to file
def save_synthetic_experiences(synthetic_experiences, filename="synthetic_experiences"):
    torch.save(f"{synthetic_experiences}.pth", filename)
    