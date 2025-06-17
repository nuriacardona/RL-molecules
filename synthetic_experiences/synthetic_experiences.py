# Imports
import argparse
import os
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
from synthetic_utils import remove_dummy_numbers, perform_action, get_reward, check_action_idx, save_synthetic_experiences
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from molecule_class import Molecule
from training_functions_mp import get_mask
import copy
import torch

########################################################################################################

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_csv', type=str, help='Name of the output file to store the experiences',
                        default='synthetic_experiences')
    parser.add_argument('--smiles', type=str, help='File containing the original SMILES from which experiences will be generated',
                        default='../data/MOSES_dataset.zip')
    parser.add_argument('--num_smiles', type=int, help='Number of smiles in "smiles" considered to generate experiences',
                        default=10000)
    parser.add_argument('--qed_target', type=float, help='Target QED value', choices=[0.7, 0.8, 0.9, 1], default=0.8)
    parser.add_argument('--fragments_file', type=str, help='Absolute or relative path to the set of fragments CSV',
                        default='../data/fragments.csv')
    parser.add_argument('--embedding_atoms', type=str, help='Atom types for the embedding process', default='C,N,O,F,S,Cl,Br')
    parser.add_argument('--addition_atoms', type=str, help='Atom types for scaffold addition', default='C,N,O,F')
    parser.add_argument('--backward_actions', type=bool, help='Allow backward actions', default=True)
    return parser.parse_args()

args = arg_parser()

########################################################################################################

# SMILES
smiles = pd.read_csv(args.smiles)
smiles = list(smiles[smiles['SPLIT']=='train']['SMILES'])

# Embedding: bond, atom and fragment types
fragments_pd = pd.read_csv(args.fragments_file, header=None)
embedding_fragmenttypes = list(fragments_pd[0])
embedding_bondtypes = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
embedding_atomtypes = args.embedding_atoms.split(',')
# Addition (molecule modification): atom and fragment types
addition_atomtypes = args.addition_atoms.split(',')
addition_fragmenttypes = list(fragments_pd[0])

# Custom set of 29 fragments
set_fragments = [remove_dummy_numbers(fragment) for fragment in list(fragments_pd[0])]

# Action tuple: done indicator
done = False # intermediate experiences
# Backward actions (i.e. whether backward/removal actions are allowed or not)
backward_actions = args.backward_actions

# Mininum reward (for min-max normalization)
target2min = {0.7: 0.7407407407407407, 0.8: 0.7142857142857143, 0.9: 0.6896551724137931, 1:0.6666666666666666} # {qed_target1: min_reward1, qed_target2, min_reward2, ...}
min_reward = target2min[args.qed_target]

# Initialize a list to store the generated experiences
experiences = []

# Iterate through the training smiles (MOSES dataset)
for s in smiles[:int(args.num_smiles)]:
    print(f"\n\nSMILES: {s}")

    # Create a molecule from the SMILES
    molecule_moses = Molecule(s, embedding_bondtypes, embedding_atomtypes, embedding_fragmenttypes, addition_atomtypes, addition_fragmenttypes)

    # BRICS decomposition
    brics_fragments = BRICSDecompose(molecule_moses.rdkit)
    print(f"    - BRICS decomposition: {brics_fragments}")
    
    # Iterate thorugh the generated BRICS fragments
    for smiles_fragment in brics_fragments:
        # Remove the dummy number (e.g. from '[3*]OC' to '[*]OC')
        smiles_fragment = remove_dummy_numbers(smiles_fragment)
        # Check if each BRICS fragment is contained within the set of fragments
        if smiles_fragment in set_fragments:
            print(f"        - Fragment {smiles_fragment} is in the set of interest")
            # Create a molecule from the fragment SMARTS
            mol_fragment = Chem.MolFromSmarts(smiles_fragment)
            # Find matches of the fragment in the MOSES molecule
            matches = molecule_moses.rdkit.GetSubstructMatches(mol_fragment)
            # Iterate through the matches
            for match in matches:
                # Atom through which the fragment is connected with the rest of the molecule
                linker_atom = match[0]
                external_bonds = 0 # Bonds between one fragment atom and one atom of the rest of the molecule
                internal_bonds = 0 # Bonds between pairs of fragment atoms
                # Iterate through the atoms inside the match
                for atom_idx in match:
                    # Retrieve the atom
                    atom = molecule_moses.rdkit.GetAtomWithIdx(atom_idx)
                    # Iterate through the bonds of the atom
                    for bond in atom.GetBonds():
                        # Retrieve the neighbor idx (the idx of the other atom involved in the bond)
                        neighbor_idx = bond.GetOtherAtom(atom).GetIdx()
                        # If the neighbor is not part of the match
                        if neighbor_idx not in match:
                            # Do not consider the linker atom (because it is allowed to have external connections)
                            if atom_idx in match[1:]:
                                external_bonds += 1
                        else:
                            # Count the fragment internal bond
                            internal_bonds += 1
                if external_bonds == 0 and (internal_bonds/2) == mol_fragment.GetNumBonds():
                        # Create a copy of the molecule
                        molecule_moses_copy = copy.deepcopy(molecule_moses)
                        # Node of the molecule to which the fragment is connected ("dummy atom")
                        node_idx = match[0]
                        if all(node_idx > idx for idx in match[1:]):
                            node_idx -= len(match[1:])
                        # Remove the atoms that belong to the fragment match
                        for atom_idx in sorted(match[1:], reverse=True):
                            molecule_moses_copy.remove_component(atom_idx)
                        # Get the data object of the molecule without the fragment ('state')
                        state = molecule_moses_copy.get_dataobj()

                        ## Create the 'action' tuple
                        # number of possible external nodes (atoms and fragments)
                        num_external = molecule_moses_copy.get_numexternal()
                        # number of possible actions that can be applied to each node (can also be thought of as the number of logits that correspond to actions involving each single node)
                        if backward_actions:
                            # If backward actions are allowed, it is necessary to add 1 to the number of external (in order to take into account the 'backward' action for each node)
                            total_actions_per_node = num_external+1
                        else:
                            # Each node can only have addition actions (no backward actions, i.e. the node can't be removed)
                            total_actions_per_node = num_external
                        action_idx = 1+(total_actions_per_node)*node_idx+len(addition_atomtypes)+set_fragments.index(smiles_fragment)
                        action = ("AddNode", {"node_idx": node_idx, "external_idx": len(addition_atomtypes)+set_fragments.index(smiles_fragment)}, action_idx)
                        print(f"        - Action: {action}")
                        num_nodes = molecule_moses_copy.get_numnodes()

                        # Perform the action
                        perform_action(molecule_moses_copy, action)
                        # Get reward
                        reward = get_reward(molecule_moses_copy, args.qed_target, min_reward)
                        print(f"        - Reward: {reward}")
                        # Get the data object of the molecule with the fragment ('next_state')
                        next_state = molecule_moses_copy.get_dataobj()
                        # Compute the mask
                        mask = get_mask(molecule_moses_copy, 2, backward_actions)
                        
                        # Check that the generated molecule is the same as the original one
                        smiles1 = Chem.MolToSmiles(molecule_moses.rdkit, canonical=True)
                        smiles2 = Chem.MolToSmiles(molecule_moses_copy.rdkit, canonical=True)
                        # Check that the generated action tuple is correct (try doing the reverse process starting from the action_idx)
                        checked_action = check_action_idx(action_idx, total_actions_per_node, num_nodes, num_external)
                        if smiles1 == smiles2 and checked_action == action:
                            experiences.append((state, action, reward, next_state, done, mask))
                            print(f"    - Experience successfully generated")

save_synthetic_experiences(experiences, args.output_csv)