import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import DNATransportHamiltonianGNN
from dataset import sequence_to_graph
from visualize_dna_graph import visualize_dna_graph
from matplotlib import pyplot as plt
import torch
from torch_geometric.data import Batch

# Generate a simple 8 BP DNA
dna_sequence = "ATCGATCG"
complementary_sequence = "GCATGCAT"
graph = sequence_to_graph(
    primary_sequence=dna_sequence, 
    complementary_sequence=complementary_sequence,
    left_contact_positions=('primary', [0]),
    right_contact_positions=('complementary', [0]),
    left_contact_coupling=0.1,
    right_contact_coupling=0.1
)
fig, ax = visualize_dna_graph(graph, primary_sequence=dna_sequence, complementary_sequence=complementary_sequence)
plt.show()

# Create the model object
model = DNATransportHamiltonianGNN()

# Run get_contact_vectors
x = graph.x
edge_index = graph.edge_index
edge_attr = graph.edge_attr
GammaL, GammaR= model.get_contact_vectors(x, edge_attr, edge_index)
print(GammaL, GammaR)

# Run NEGFProjector using a dummy hamiltonian
dummy_hamiltonian = torch.randn(model.num_unique_elements) 
dos_pred, transmission_pred, H = model.NEGFProjection(dummy_hamiltonian, GammaL, GammaR)
print(dos_pred, transmission_pred)
print(H)

# Run forward
dos_pred, transmission_pred = model.forward(graph)

