import pandas as pd
import networkx as nx
from itertools import combinations

df = pd.read_csv("processed/train.csv")

G = nx.Graph()

# add nodes
G.add_nodes_from(df["TransactionID"])

# edges (optimized)
for col in ["card1", "addr1", "P_emaildomain"]:
    for _, group in df.groupby(col):
        ids = group["TransactionID"].values
        if len(ids) > 1:
            G.add_edges_from(combinations(ids, 2))