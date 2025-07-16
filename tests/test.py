from dna_transport_gnn import DNATransportGNN, DNATransportDataset, DNASequenceToGraph
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
A = DNASequenceToGraph()
seq = A.sequence_to_graph('ACGC__', '__GCGT', 
    left_contact_positions=('primary',[0, 1]), 
    right_contact_positions=('complementary',[4, 5]),   
    left_contact_coupling=0.1, 
    right_contact_coupling=0.1
)
nx.draw(to_networkx(seq), with_labels=True)
plt.show()