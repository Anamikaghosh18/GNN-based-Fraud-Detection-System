import torch
from torch_geometric.data import HeteroData
import numpy as np
import pandas as pd


class GraphBuilder:
    def __init__(self):
        self.uid_map = None  # store mapping (important later)

    def construct_hetero_graph(self, df):
        data = HeteroData()

        # 1. Transaction Features
        feature_cols = [c for c in df.columns if c not in ['isFraud', 'TransactionID', 'uid', 'TransactionDT']]

        # Ensure numeric
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
        df[feature_cols] = df[feature_cols].fillna(0)

        x = df[feature_cols].values.astype(np.float32)

        data['transaction'].x = torch.tensor(x, dtype=torch.float32)
        data['transaction'].num_nodes = len(df)

        # 2. Labels
        if 'isFraud' in df.columns:
            data['transaction'].y = torch.tensor(df['isFraud'].values, dtype=torch.long)

        # 3. User Nodes (uid mapping)
        if self.uid_map is None:
            self.uid_map = {uid: i for i, uid in enumerate(df['uid'].unique())}

        df['uid_idx'] = df['uid'].map(self.uid_map)

        user_indices = torch.tensor(df['uid_idx'].values, dtype=torch.long)
        transaction_indices = torch.arange(len(df))

        num_users = len(self.uid_map)

        # user node features (simple embedding placeholder)
        data['user'].x = torch.zeros((num_users, 1))  # minimal feature
        data['user'].num_nodes = num_users

        # 4. Edges
        edge_index = torch.stack([transaction_indices, user_indices], dim=0)

        data['transaction', 'belongs_to', 'user'].edge_index = edge_index

        data['user', 'performs', 'transaction'].edge_index = torch.stack([
            user_indices,
            transaction_indices
        ], dim=0)

        return data