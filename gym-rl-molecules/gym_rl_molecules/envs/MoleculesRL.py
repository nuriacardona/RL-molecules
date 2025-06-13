## Imports
# General tools
import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
# RDkit
#from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.Descriptors import qed
from rdkit.Chem import Draw
# Gymnasium
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
# Molecule class
from .molecule_class import Molecule

# Seeds
np.random.seed(42)
random.seed(42)


# CUSTOM RL ENVIRONMENT
class MolecEnv(gym.Env):
    def __init__(self):
        super(MolecEnv, self).__init__()

        #### ACTION SPACE -> composite (4 components: episode_term, node1, node2, edge)
        self.action_space =  spaces.Discrete(250)
        #### OBSERVATION SPACE (STATE OF THE MOLECULE)
        self.observation_space = spaces.Graph(node_space=Box(low=-1, high=float('inf')), edge_space=Box(low=-1, high=float('inf')))

    def initialize_parameters(self,
                 embedding_atomtypes,
                 embedding_bondtypes,
                 embedding_fragmenttypes,
                 addition_atomtypes,
                 addition_fragmenttypes,
                 ls_scaffolds,
                 max_episode_steps,
                 qed_target,
                 min_reward,
                 render_molecules,
                 render_mode):
        """
        Initialize all environment parameters. Most of them are required to initialize 
            the molecule at the beginning of each episode.
        
        Args:
            embedding_atomtypes (list): atom types (categories) that can be encountered 
                during the embedding process
            embedding_bondtypes (list): bond types (categories) that can be encountered 
                during the embedding process
            embedding_fragmenttypes (list): fragment types (categories) that can be 
                encountered during the embedding process
            addition_atomtypes (list): atom types that can be used for scaffold addition
            addition_fragmenttypes (list): fragment types that can be used for scaffold 
                addition
            ls_scaffolds (list): possible scaffolds for the initial state (S0)
            max_episode_steps (int): Maximum number of episode steps
            qed_target (float): QED target value during the generation process
            min_reward (float): Minimum reward value
            render_molecules (boolean): Whether to render or not the molecule at each 
                step
            render_mode (str): Type of rendering
        """
        self.embedding_atomtypes = embedding_atomtypes
        self.embedding_bondtypes = embedding_bondtypes
        self.embedding_fragmenttypes = embedding_fragmenttypes
        self.addition_atomtypes = addition_atomtypes
        self.addition_fragmenttypes = addition_fragmenttypes
        self.ls_scaffolds = ls_scaffolds
        self.max_episode_steps = max_episode_steps
        self.qed_target = qed_target
        self.min_reward = min_reward
        self.render_molecules = render_molecules
        self.render_mode = render_mode

    def _reset_molecule(self):
        """
        Create a molecule instance
        """
        # Randomly select an initial state from the possible scaffolds
        initial_scaffold = random.choice(self.ls_scaffolds)
        # Initialize a molecule with all required parameters
        self.Mol = Molecule(initial_scaffold,
                            self.embedding_bondtypes, 
                            self.embedding_atomtypes,
                            self.embedding_fragmenttypes,
                            self.addition_atomtypes,
                            self.addition_fragmenttypes)
    
    def _get_obs(self):
        """
        Get an observation of the current state of the molecule.

        Returns:
            Data object representing the molecule's state.
        """
        data_obj = self.Mol.get_dataobj()
        return data_obj

    def _get_info(self):
        """
        Retrieve information about the current number of atoms in the molecule

        Returns:
            int indicating the total cound of atoms
        """
        return self.Mol.get_numatoms()
    
    def _get_Mol_copy(self):
        """
        Create a copy of the molecule

        Returns:
            Deep copy of the current molecule inside the environment
        """
        return copy.deepcopy(self.Mol)
    
    def render(self):
        """
        Visualization of the environment state.

        Returns:
            Array representing the image or display of a matplotlib figure 
        """
        # Activate the display of atom indices
        dos = Draw.MolDrawOptions()
        dos.addAtomIndices=True
        # Convert the rdkit molecule to an image
        img = Draw.MolToImage(self.Mol.rdkit, options=dos, size=(300, 300))
        # Depending on the render mode selected, output an image or an RGB array.
        if self.render_mode == 'human':
            # matplotlib image
            plt.figure(figsize=(6,6))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            plt.close()
        else:
            # RGB array
            return np.asarray(img)

    def get_reward(self, qed_mol):
        """
        Compute the reward associated to the current environment state (molecule)

        Args:
            qed_mol (float): QED score of the modified molecule.
        
        Returns
            float value indicating the reward associated to the input QED score.
        """
        alpha = 0.5
        reward = 1/(1+alpha*abs(qed_mol-self.qed_target))
        # Min-max normalization [0,1]
        max_reward = 1.0
        scaled_reward = (reward - self.min_reward) / (max_reward - self.min_reward)

        return scaled_reward

    def validate_action(self):
        """
        Validate that the performed action leads to a chemically valid molecule.

        Returns:
            boolean indicating whether the molecule has a feasible valency and
                is chemically valid
        """
        if self.Mol.check_molecule():
            return True
        return False
    
    def reset(self):
        """
        Set the environment to the initial state (S0). Reset all the episode and 
            state-based variables in our environment

        Returns:
            Data object representing the molecule's state (observation) and
                dictionary with molecular information
        """

        # Reset the molecule present inside the environment
        self._reset_molecule()
        # Retrieve the observation of the new molecule
        observation = self._get_obs()
        # Construct a dictionary with essential information about the new molecule
        info = {'mol_copy': self._get_Mol_copy(), 'smiles': self.Mol.get_molecsmiles(), 'Num_atoms': self._get_info(), 'Num_nodes': self.Mol.get_numnodes()}

        # Initialize all lists
        self.ls_actions = []
        self.ls_acceptance = []
        self.ls_step_rewards = []
        self.ls_molecules = []
        self.ls_tpsa = []
        self.ls_mw = []
        self.ls_logp = []
        self.ls_qed = []

        # Compute and store initial molecular properties
        tpsa_initial, mw_initial, logp_initial, qed_initial = self.Mol.get_molecularscores()
        self.ls_tpsa.append(tpsa_initial)
        self.ls_mw.append(mw_initial)
        self.ls_logp.append(logp_initial)
        self.ls_qed.append(qed_initial)
        # Store the initial molecule (rdkit)
        self.ls_molecules.append(Chem.Mol(self.Mol.rdkit))

        # Rendering of the molecule
        if self.render_molecules:
            self.render() 

        return observation, info

    def step(self, input_action):
        """
        Take a step (action) within the environment

        Args:
            input_action (tuple): action type, action details, action
                index and step number.
        
        Returns:
            Data object representing the molecule's state (observation),
                float indicating the reward associated to the performed
                action, boolean indicating if a terminal state was
                reached, boolean indicating if the a custom limit stops
                the episode and dictionary with information about the 
                molecule.
        """

        ############################################################
        # Perform the chosen modification (action) in the molecule #
        ############################################################

        #### Old (original) molecule
        # Keep a copy
        old_Molecule = copy.deepcopy(self.Mol)
        num_atoms = self.Mol.rdkit.GetNumAtoms()


        #### Action (modification) to be performed in the molecule
        action_type, action_details, action_idx, step = input_action

        # ADDNODE ACTION: EXTERNAL ATOM/FRAGMENT ADDITION
        if action_type == "AddNode":
            # NODE 1 INDEX IDENTIFICATION: Existing node in the molecule
            nx_idx_1 = action_details["node_idx"] # nxgraph index

            # NODE 2 INDEX IDENTIFICATION: External atom/fragment to be added to the molecule
            external_idx = action_details["external_idx"] # nxgraph index
            # a) Add Atom
            if external_idx < len(self.addition_atomtypes):
                # Retrieve the atom type that will be added
                atom = self.addition_atomtypes[external_idx]
                # Add the atom to the molecule
                nx_idx_2 = self.Mol.add_atom(atom)
                # Not necessary to remove an index
                rdkit_idx_remove = None
            # b) Add Fragment
            else:
                # Retrieve the fragment type that will be added
                fragment = self.addition_fragmenttypes[external_idx-len(self.addition_atomtypes)]
                # Add the fragment to the molecule
                nx_idx_2, rdkit_idx_remove = self.Mol.add_fragment(fragment)
        # REMOVENODE ACTION: ELIMINATE AN EXISTING ATOM/FRAGMENT
        elif action_type == "RemoveNode":
            # Retrieve the index of the node to be removed
            removed_nx_idx = action_details["node_idx"]
            # Remove the node
            self.Mol.remove_component(removed_nx_idx)
        # ADDEDGE ACTION: NON-EDGE IDENTIFICATION
        elif action_type == "AddEdge":
            # Retrieve the index of the non-edge to be created
            nonedge_idx = action_details["nonedge_idx"]
            # Get the list of non-edges from the molecule
            nonedges_nxgraph = self.Mol.get_nonedges()
            # Select the non-edge of interes
            target_nxnonedge = nonedges_nxgraph[nonedge_idx]
            # Index of the first node involved in the non-edge
            nx_idx_1 = target_nxnonedge[0].item()
            # Index of the second node involved in the non-edge
            nx_idx_2 = target_nxnonedge[1].item()
            # Not necessary to remove an index
            rdkit_idx_remove = None

        # ADDNODE AND ADDEDGE ACTIONS: BOND CREATION
        if (action_type == "AddNode") or (action_type == "AddEdge"):
            # All newly created bonds are single
            bond_type = Chem.BondType.SINGLE
            # Create the single bond betweeb the nodes previously established 
            new_bond = self.Mol.add_bond(nx_idx_1, nx_idx_2, rdkit_idx_remove, bond_type)
            # Save the arguments of the performed action
            args = (action_type, action_details, bond_type)
            self.ls_actions.append(args)

        # STOP ACTION: SET EPISODE TERMINATION TO TRUE
        if (action_type == "Stop") or (step == self.max_episode_steps):
            # Indicate that the environment arrives at a terminal state
            episode_term = True
            # Save the arguments of the performed action
            self.ls_actions.append(("Stop", None))
        else:
            # Indicate that the environment is not in a terminal state
            episode_term = False


        ##########
        # Reward #
        ##########
        # Action acceptance
        if action_type in ["AddNode", "AddEdge", "RemoveNode"]:
            # Check if the performed action was valid
            action_accepted = self.validate_action()
            if action_accepted:
                # Compute molecular scores of the valid molecule
                tpsa, mw, logp, qed_mol = self.Mol.get_molecularscores()
            else:
                # Set all molecular properties to -1
                tpsa, mw, logp, qed_mol = (-1, -1, -1, -1)
                # Restore the previous state of the molecule (before last modification)
                self.Mol = old_Molecule
        else: # Stop action
            action_accepted = True
            # Compute molecular scores
            tpsa, mw, logp, qed_mol = self.Mol.get_molecularscores()

        # Compute the reward
        reward = self.get_reward(qed_mol)

        # Store all the information regarding the action that was performed
        self.ls_acceptance.append(action_accepted)
        self.ls_molecules.append(Chem.Mol(self.Mol.rdkit))
        self.ls_step_rewards.append(reward)
        self.ls_tpsa.append(tpsa)
        self.ls_mw.append(mw)
        self.ls_logp.append(logp)
        self.ls_qed.append(qed_mol)

        ## Prepare the return of the episode step
        Truncated = False
        # Get the observation of the current state of the environment (molecule)
        observation = self._get_obs()
        # Only provide all the episode information if it's the last step
        if episode_term == True:
            info = {'Actions': self.ls_actions, 
                    'Acceptance': self.ls_acceptance, 
                    'StepRewards': self.ls_step_rewards, 
                    'Molecules': self.ls_molecules, 
                    'TPSA': self.ls_tpsa, 
                    'MW': self.ls_mw, 
                    'LogP': self.ls_logp, 
                    'QED': self.ls_qed, 
                    'Num_atoms': self._get_info(),
                    'Num_nodes': self.Mol.get_numnodes(),
                    'last_TPSA': tpsa, 
                    'last_MW': mw, 
                    'last_LogP': logp, 
                    'last_QED': qed_mol,
                    'action_accepted': action_accepted,
                    'mol_copy': self._get_Mol_copy(),
                    'smiles': self.Mol.get_molecsmiles()}
        # Step-wise episode: provide only reduced information
        else:
            info = {'TPSA': tpsa, 
                    'MW': mw, 
                    'LogP': logp, 
                    'QED': qed_mol, 
                    'Num_atoms': self._get_info(),
                    'Num_nodes': self.Mol.get_numnodes(),
                    'action_accepted': action_accepted, 
                    'mol_copy': self._get_Mol_copy(),
                    'smiles': self.Mol.get_molecsmiles()}
        
        # Rendering of the molecule
        if self.render_molecules:
            self.render() 
        
        return observation, reward, episode_term, Truncated, info
