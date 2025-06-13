## Imports
# Pytorch and Pytorch Geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_scatter import scatter_add
#seeds
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# NEURAL NETWORK (NN) CLASS (ACTION-VALUE FUNCTION)
class NN_action_value(nn.Module): # ACTOR
    def __init__(self, num_categories, size_emb, hidden_dim, types_edges, num_external, backward_actions=True):
        """Input parameters:
        
        Args:
            num_categories: number of different types of atoms the 
                initial molecule can contain (coincides with the 
                number of atoms in the "dict_atoms")
            size_emb: size of the embedding vectors
            num_atoms: number of atoms the molecule has
            types_atoms_and_bonds: types of atoms/bonds we have in 
                our atom/bond pool (e.g. C, N, O, F (atoms) and 
                single, double, triple, arotmatic (bonds) would be 
                types_atoms_and_bonds = 4)
            types_actions: how many actions we have in our action
                 space (e.g. add atom, remove atom, replace atom, 
                 add bond, replace bond would be types_action = 5)
            num_external: number of total external atoms and fragments
                that can be added (i.e. 0 = C, 1 = N, 2 = O, 3 = F, 
                4 = [9*]n1nnnc1N, 5 = [9*]n1nnc2sc3c(c2c1=O)CCCC3, ...)
        """

        super(NN_action_value, self).__init__()
        
        # Embedding layers
        self.node_emb = nn.Embedding(num_categories, size_emb) # Node embedding
        self.edge_emb = nn.Embedding(types_edges, size_emb) # Edge embedding
        
        # GATConv layers (node embedding)
        self.conv1 = GATConv(size_emb, hidden_dim, edge_dim=size_emb)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=size_emb)

        # MLPs for the different action types
        # Stop
        self.mlp_stop = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
        # AddNode
        if backward_actions:
            # add 1 to the number of external in order to consider the backward action (reverse)
            output_dimension_addnode = num_external+1
        else:
            output_dimension_addnode = num_external
        self.mlp_addnode = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dimension_addnode)
                )
        # AddEdge
        self.mlp_addedge = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
    

    def forward(self, inp_batch, state=None, **kwargs):
        """Forward pass to obtain the logits (q-values)
        
        Args:
            inp_batch: input batch of transitions
        
        Returns:
            q-values of each instance inside the input batch
        """

        #### (0) Inputs
        x = inp_batch.x # Node features
        edge_index = inp_batch.edge_index # Connectivity of the graphs
        edge_attr = inp_batch.edge_attr # Edge features
        batch_tensor = inp_batch.batch # Batch information
        non_edges = inp_batch.non_edges # Non-edges of the molecule
        batch_non_edge = inp_batch.batch_non_edge # Batch information regarding the non-edges
        ptr = inp_batch.ptr # offsets between batches

        #### (1) Embedding
        ## (1.1) MOLECULE EDGES
        if edge_attr is not None: # specifically check it is different than 'None' to avoid an error when working with batch_size > 1.
            edge_attr = self.edge_emb(edge_attr)
        ## (1.1) MOLECULE NODES
        # Embedding
        node_embeddings = self.node_emb(x).squeeze(1)
        # GCN layers (node embedding)
        node_embeddings = self.conv1(node_embeddings, edge_index, edge_attr)
        node_embeddings = F.relu(node_embeddings)
        node_embeddings = self.conv2(node_embeddings, edge_index, edge_attr)
        # Summed node embeddings
        summed_node_embeddings = scatter_add(node_embeddings, batch_tensor, dim=0)

        #### (2) Predict logits for each action
        ## (2.1) Stop
        stop_logits = self.mlp_stop(summed_node_embeddings)
        ## (2.2) AddNode
        addnode_logits = self.mlp_addnode(node_embeddings)
        ## (2.3) AddEdge
        # Sum the embeddings of the 2 nodes involved in each non-edge
        node_indices_1 = non_edges[:, 0]
        node_indices_2 = non_edges[:, 1]
        embeddings_1 = node_embeddings[node_indices_1]
        embeddings_2 = node_embeddings[node_indices_2]
        non_edges_embeddings = embeddings_1 + embeddings_2
        # MLP
        addedge_logits = self.mlp_addedge(non_edges_embeddings)

        #### (3) Padding: to the dimension of the largest instance in the batch        
        ## (3.1) Split stop per batch
        stop_grouped = torch.unbind(stop_logits)
        ## (3.2) Split nodes per batch
        nodes_grouped = torch.split(addnode_logits, torch.bincount(batch_tensor).tolist())
        ## (3.3) Split non-edges per batch (specify 'minlength' to cover the caser in which the last batch instance(s) do not have non-edges)
        edges_grouped = torch.split(addedge_logits, torch.bincount(batch_non_edge, minlength=batch_tensor.unique().size(0)).tolist())
        ## (3.4) Concatenate the logits of each batch instance in a vectorized way
        raw_logits = [torch.cat([stop, nodes.flatten(), edges.flatten()]) if len(edges) > 0 
                else torch.cat([stop, nodes.flatten()]) for stop, nodes, edges in 
                zip(stop_grouped, nodes_grouped, edges_grouped + ((),))]
        ## (3.5) Zero padding
        padded_logits = torch.nn.utils.rnn.pad_sequence(raw_logits, batch_first=True)

        #### Return of the forward pass
        return padded_logits.squeeze()
