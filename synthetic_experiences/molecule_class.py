## Imports
# General tools
import copy
import networkx as nx
import os
import sys
# RDkit
#from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.Descriptors import qed, MolLogP, TPSA, ExactMolWt, NumRadicalElectrons
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# Pytorch
import torch
from torch_geometric.utils.convert import from_networkx

# Seeds
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


# CUSTOM MOLECULE CLASS (ENVIRONMENT STATE)
class Molecule:
    """
    Molecule class to represent the environment state (molecule being constructed).
    The molecule is simultaneously represented as a Network X graph (fragment level representation) and an editable RDKit molecule (atom level representation).
    """

    def __init__(self, smiles, embedding_bondtypes, embedding_atomtypes, embedding_fragmenttypes, addition_atomtypes, addition_fragmenttypes):
        """Initialize a molecule instance from a SMILES string and 
        
        Args:
            smiles (str): SMILES string of the initial molecule
            embedding_bondtypes (list of str): bond types that can be present in 
                the molecule
            embedding_atomtypes (list of str): atom types that can be present in 
                the molecule
            embedding_fragmenttypes (list of str): fragment types that can be 
                present in the molecule
            addition_atomtypes (list of str): atom types that can be added to the
                 molecule during generation
            addition_fragmenttypes (list of str): fragment types that can be added
                 to the molecule during generation

        Returns:
            None
        """

        #### Mapping from molecule component (atom or fragment, e.g. 'C', 'N', '[1*]C(C)=O', etc.) to feature (nº to embed that component, e.g. 0, 1, 2)
        self.embedding_comp2feat = {} # dictionary 'molecule component (key) to feature (value)', e.g. {'C': 0, 'N': 1, '[1*]C(C)=O': 2, ...}
        ## Fill the dictionary with the input atom, fragment and bond types
        idx = 0
        for atom in embedding_atomtypes:
            self.embedding_comp2feat[atom] = idx
            idx += 1
        for fragment in embedding_fragmenttypes:
            self.embedding_comp2feat[fragment] = idx
            idx +=1
        # Index is reset to 0 regarding bond types (edge features)
        idx = 0
        for bond in embedding_bondtypes:
            self.embedding_comp2feat[bond] = idx
            idx += 1
        self.addition_atomtypes = addition_atomtypes
        self.addition_fragmenttypes = addition_fragmenttypes
        self.ls_external = addition_atomtypes + addition_fragmenttypes
        self.num_external = len(self.ls_external)

        #### Mapping from nxgraph node idx to rdkit molecule atom idx
        self.d_nxidx2rdkitidx = {}

        #### 2 representations of the molecule
        ## 1) rdkit: atom level (RWMol is a molecule class intended to be edited (Read-Write molecule))
        self.rdkit = Chem.RWMol(Chem.MolFromSmiles(smiles))
        ## 2) networkx graph: fragment level
        self.nxgraph = self.rdkitmol_to_nxgraph()


    def rdkitmol_to_nxgraph(self):
        """Create a NetworkX graph representation from an RDKit molecule.
        
        Returns:
            networkx.Graph: NetworkX graph representing the input molecule.
                Nodes correspond to atoms, edges correspond to bonds. Attributes
                extracted from the embedding_comp2feat dictionary.
        """
        
        # Empty NetworkX graph
        G = nx.Graph()

        # Add nodes (atoms) with feature embeddings as attributes
        for atom in self.rdkit.GetAtoms():
            # The nxgraph index of the node repesenting the atom coincides with the RDKit index
            nx_idx = atom.GetIdx()
            # Create the node in the NetworkX graph
            G.add_node(
                nx_idx,
                x = self.embedding_comp2feat[atom.GetSymbol()]
            )
            # Update the mapping from nxgraph node idx to rdkit molecule atom idx
            self.d_nxidx2rdkitidx[nx_idx] = {'linker': atom.GetIdx(), 'all': [atom.GetIdx()], 'SMILES': atom.GetSymbol()}
        
        # Add edges (bonds) with feature embeddings  attributes
        for bond in self.rdkit.GetBonds():
            # Create the edge in the NetworkX graph
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                edge_attr = self.embedding_comp2feat[bond.GetBondType()]
            )

        # If the molecule has no edges and only 1 atom (e.g. 'C'), add a self loop (requirement for the forward pass)
        if (not list(G.edges)) and (self.rdkit.GetNumAtoms() == 1):  # Single atom, no bonds
            # Add a self-loop
            G.add_edge(0,
                       0, 
                       edge_attr = self.embedding_comp2feat[Chem.BondType.SINGLE])
        return G

    def get_dataobj(self, get_non_edges=True):
        """Create a data object representing the molecule at the fragment level
        
        Args:
            get_non_edges (optional; boolean, default is True): boolean 
                indicating whether to include or not the list of non-edges 
                of the nxgraph.
        
        Output:
            torch_geometric.data.Data object created from the nxgraph of
                 the molecule 
        """

        # Generate the data object from the NetworkX graph
        data = from_networkx(self.nxgraph)
        # Add non-edges (if necessary)
        if get_non_edges:
            data.non_edges = self.get_nonedges()
        return data

    def get_nonedges(self):
        """Compute and return the list of non-edges of the current molecule graph (nxgraph)
        
        Returns:
            Tensor containing the set of undirected non-edges
        """

        return torch.tensor(list(nx.non_edges(self.nxgraph)), dtype=torch.long)
    
    def get_molecularscores(self):
        """Compute and return the scores of the molecule
        
        Returns:
            Tuple containing the 4 molecular scores (tpsa, mw, logp, qed)
        """

        # Sanitize the molecule
        Chem.SanitizeMol(self.rdkit)
        # Compute the molecular properties
        try:
            tpsa = TPSA(self.rdkit)
            mw = ExactMolWt(self.rdkit)
            logp = MolLogP(self.rdkit)
            qed_val = qed(self.rdkit)
            return (tpsa, mw, logp, qed_val)
        except:
            # Set all properties to -1 if it is not possible to compute the scores from the current molecule (invalid)
            return (-1, -1, -1, -1)

    def get_molecsmiles(self):
        """Get the molecule's SMILES string
        
        Returns:
            str corresponding to the SMILES string of the molecule
        """

        # Generate the SMILES string from the RDKit representation of the molecule
        smiles = Chem.MolToSmiles(self.rdkit)
        return smiles
        
    def get_numatoms(self):
        """Compute the number of atoms of the rdkit molecule
    
        Returns:
            int representing the number of atoms of the rdkit molecule
        """

        return self.rdkit.GetNumAtoms()
    
    def get_numnodes(self):
        """Compute the number of nodes of the nxgraph
        
        Returns:
            int representing the number of nodes in the nxgraph
        """

        return self.nxgraph.number_of_nodes()

    def get_numexternal(self):
        """Return the number of atoms and fragments (globally known as 'external') that can be added to the molecule
        
        Returns:
            int representing the number of external (atoms and fragments)
        """

        return self.num_external
        
    def add_atom(self, atom):
        """Add the input atom and update both representations of the molecule
            - In the rdkit representation, the new atom will be an atom
            - In the networkx graph representation, the new atom will be a single node

        Args:
            atom (str): element symbol of the atom to be added (e.g. 'C')
        
        Returns:
            nx_idx (int): index of the networkx graph (nxgraph) node representing the added atom
        """

        #### Update the 2 representations of the molecule
        ## 1) rdkit
        rdkit_idx = self.rdkit.AddAtom(Chem.Atom(atom))
        ## 2) networkx graph
        # Assign the next available index to the atom (indices start at 0)
        nx_idx = self.nxgraph.number_of_nodes()
        # Create the node in the NetworkX graph
        self.nxgraph.add_node(
            # Assign the index of the nxgraph node
            nx_idx,
            # Assign the embedding number of the atom
            x = self.embedding_comp2feat[atom]
        )

        # Udate the mapping from nxgraph node idx to rdkit molecule atom idx
        self.d_nxidx2rdkitidx[nx_idx] = {'linker':rdkit_idx, 'all':[rdkit_idx], 'SMILES':self.rdkit.GetAtomWithIdx(rdkit_idx).GetSymbol()}

        return nx_idx

    def remove_component(self, removed_nx_idx):
        """Remove a component of the molecule (atom/fragment) and update both representations of the molecule
        
        Args:
            removed_nx_idx (int): index of the molecule component to be removed
        """

        #### (0) COMPONENT INFORMATION
        ## Retrieve the rdkit atom indices constituting the component (atom/fragment) to be removed
        rdkit_all = self.d_nxidx2rdkitidx[removed_nx_idx]['all']
        ## Number of rdkit atoms that will be deleted (after all removals, the indices of the remaining atoms in the rdkit molecule can potentially change)
        num_idx_shifts = len(rdkit_all)
        ## Select the highest rdkit index that has been removed
        highest_removed_rdkit_idx = max(rdkit_all)

        #### (1) RDKIT
        ## Remove the rdkit atoms and all their bonds (in descending order to avoid rdkit idx inconsistencies)
        for rdkit_idx in sorted(rdkit_all, reverse=True):
            self.rdkit.RemoveAtom(rdkit_idx)
        
        #### (2) NXGRAPH
        ## Remove the node and all of its edges
        self.nxgraph.remove_node(removed_nx_idx)
        ## Update the necessary nxgraph indices
        # Initialize a dictionary to map old to new indices in nxgraph {old_idx: 'new_idx', old_idx: new_idx, etc.}
        nx_idx_mapping = {}
        # Iterate through nxgraph indices
        for nx_idx in self.nxgraph.nodes:
            # Nodes with indices greater than the removed one will have their original index subtracted by 1 (only 1 networkx index has been removed)
            if nx_idx > removed_nx_idx:
                # Subtract 1 (only 1 networkx node has been removed)
                nx_idx_mapping[nx_idx] = nx_idx - 1
        # Re-label the necessary nxgraph nodes
        if nx_idx_mapping:
            nx.relabel_nodes(self.nxgraph, nx_idx_mapping, copy=False)
        
        #### (3) DICTIONARY 'd_nxidx2rdkitidx'
        # Remove the dictionary entry of the eliminated component
        del self.d_nxidx2rdkitidx[removed_nx_idx]
        # Update the dictionary key values (nxgraph indices)
        self.d_nxidx2rdkitidx = {new_idx: value for new_idx, (_, value) in enumerate(sorted(self.d_nxidx2rdkitidx.items()))}
        # Update the necessary rdkit indices in the dictionary: atoms with indices greater than the removed atom have their original indices subtracted by 'num_idx_shifts'
        for nx_idx in self.d_nxidx2rdkitidx:
            if self.d_nxidx2rdkitidx[nx_idx]['linker'] > highest_removed_rdkit_idx:
                self.d_nxidx2rdkitidx[nx_idx]['linker'] = self.d_nxidx2rdkitidx[nx_idx]['linker'] - num_idx_shifts
            for rdkit_idx in range(len(self.d_nxidx2rdkitidx[nx_idx]['all'])):
                if self.d_nxidx2rdkitidx[nx_idx]['all'][rdkit_idx] > highest_removed_rdkit_idx:
                    self.d_nxidx2rdkitidx[nx_idx]['all'][rdkit_idx] = self.d_nxidx2rdkitidx[nx_idx]['all'][rdkit_idx] - num_idx_shifts

    def add_fragment(self, fragment):
        """Add the input fragment and updates both representations of the molecule.
            - In the rdkit representation, the new fragment will be represented atom by atom
            - In the networkx graph representation, the whole new fragment will be a single node
        
        Args:
            fragment (string): smiles string of the fragment (e.g. "[1*]C(C)=O").
        
        Returns:
            nx_id (int): index of the networkx graph (nxgraph) node representing the added fragment
            rdkit_idx_remove (int): index of the rdkit dummy atom, which will need to be removed after bond creation
        """

        #### Update the 2 representations of the molecule
        ## 1) rdkit
        # Create an RDKit molecule object of the fragment
        fragment_mol = Chem.MolFromSmiles(fragment)
        # Store the number of atoms in the fragment
        num_atoms_fragment = fragment_mol.GetNumAtoms() - 1 # subtract one not to consider the dummy atom
        # Store the number of atoms in the existing rdkit molecule
        num_atoms_molecule = self.rdkit.GetNumAtoms()
        # Combine the existing rdkit molecule and the newly created fragment
        self.rdkit = Chem.RWMol(Chem.CombineMols(self.rdkit, fragment_mol))
        ## 2) networkx graph
        # Assign the next availabe index to the fragment (indices start at 0)
        nx_idx = self.nxgraph.number_of_nodes()
        # Create the node in the NetworkX graph
        self.nxgraph.add_node(
            # Assign the index of the nxgraph node
            nx_idx,
            # Assign the embedding number of the fragment
            x = self.embedding_comp2feat[fragment]
        )

        # Identify dummy atom of the fragment to be removed
        for atom in self.rdkit.GetAtoms():
            if atom.GetSymbol() == "*":
                rdkit_idx_remove = atom.GetIdx()
                break
        # Identify neighbors of the dummy atom (node 2: node to establish the bond with)
        neighbor_idx = [neigh.GetIdx() for neigh in self.rdkit.GetAtomWithIdx(rdkit_idx_remove).GetNeighbors()][0]
        # Udate the mapping from nxgraph node idx to rdkit molecule atom idx
        self.d_nxidx2rdkitidx[nx_idx] = {'linker': neighbor_idx, 'all':[idx for idx in range(num_atoms_molecule, num_atoms_molecule+num_atoms_fragment)], 'SMILES':fragment}

        return nx_idx, rdkit_idx_remove

    def add_bond(self, nx_idx_1, nx_idx_2, rdkit_idx_remove = None, bond_type = Chem.BondType.SINGLE):
        """Create a bond of type 'bond_type' between the nxgraph nodes with indices 'nx_idx_1' and 'nx_idx_2'.
                The rdkit representation of the molecule is also updated accordingly (using the  mapping 'd_nxidx2rdkitidx').
        
        Args:
            x_idx_1 (int): networkx graph (nxgraph) index of the first atom involved in the bond
            nx_idx_2 (int): networkx graph (nxgraph) index of the second atom involved in the bond
            rdkit_idx_remove (optional; int or None, default is None): rdkit index of the atom to 
                be removed (only used if nx_idx_2 belongs to a fragment)
            bond_type (Chem.BondType, default is Chem.BondType.SINGLE): type of bond to be created
                 between node1 and node2
                 
        Output:
        · Boolean: boolean indicating whether the new bond was successfully created (True) or not (False)
        """

        # Avoid creating a self-loop (i.e. an atom that is bound to itself)
        if nx_idx_1 != nx_idx_2:
            #### Update the 2 representations of the molecule
            ## 1) rdkit
            # Map from networkx graph (nxgraph) indices to rdkit indices
            rdkit_idx_1 = self.d_nxidx2rdkitidx[nx_idx_1]['linker']
            rdkit_idx_2 = self.d_nxidx2rdkitidx[nx_idx_2]['linker']
            # Retrieve the bond between both atoms
            bond = self.rdkit.GetBondBetweenAtoms(rdkit_idx_1, rdkit_idx_2)
            # Avoid forming a bond that already exists
            if bond is None:
                try:
                    # Create the bond between node1 and node2
                    self.rdkit.AddBond(rdkit_idx_1, rdkit_idx_2, bond_type)
                    # Check aromaticity
                    if bond_type == Chem.BondType.AROMATIC:
                        self.rdkit.GetAtomWithIdx(rdkit_idx_1).SetIsAromatic(True)
                        self.rdkit.GetAtomWithIdx(rdkit_idx_2).SetIsAromatic(True)
                    Chem.FastFindRings(self.rdkit)
                    # Remove 'rdkit_idx_remove' if necessary
                    if rdkit_idx_remove:
                        # Remove the atom
                        self.rdkit.RemoveAtom(rdkit_idx_remove)
                        # Update d_nxidx2rdkitidx (rdkit indices change when 1 atom is removed)
                        if rdkit_idx_2 > rdkit_idx_remove:
                            # atoms with indices greater than the removed atom have their original indices subtracted by 1
                            self.d_nxidx2rdkitidx[nx_idx_2]['linker'] = rdkit_idx_2 - 1
                except:
                    return False
                else:
                    ## 2) networkx graph: only updated if the bond was successfully created
                    self.nxgraph.add_edge(
                        nx_idx_1,
                        nx_idx_2,
                        edge_attr=self.embedding_comp2feat[bond_type]
                    )
                    return True
        return False

    def check_molecule(self):
        """Check if the molecule is valid
        
        Returns:
            boolean indicating if the molecule is valid (True) or not (False)
        """

        # Try to perform a sanitize of the molecule and convert it to smiles string and back to a molecule
        try:
            Chem.SanitizeMol(self.rdkit)
            smiles_conversion = Chem.MolToSmiles(self.rdkit, isomericSmiles=True)
            back_molecule = Chem.MolFromSmiles(smiles_conversion)
            return True
        except:
            return False