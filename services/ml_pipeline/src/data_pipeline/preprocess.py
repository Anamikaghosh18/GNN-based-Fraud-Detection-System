import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

class Preprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=30)
        self.cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']

    def transform(self, df):
        # 1. Drop high-null columns (> 90% missing)
        null_pct = df.isnull().sum() / len(df)
        cols_to_drop = null_pct[null_pct > 0.90].index
        df = df.drop(columns=cols_to_drop)

        # 2. Fill NaNs
        # Using -999 for numerics helps the GNN learn that 'missing' is a specific state
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(-999)

        # 3. Cyclical Time Features
        df['hour'] = (df['TransactionDT'] // 3600) % 24
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)

        # 4. Label Encoding for Categoricals
        for col in self.cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                # Categorical missingness is handled by string conversion
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le # Store for inference later

        # 5. PCA for V-columns
        v_cols = [c for c in df.columns if c.startswith('V')]
        if v_cols:
            v_pca_data = self.pca.fit_transform(df[v_cols])
            # Create readable names for PCA columns
            pca_df = pd.DataFrame(
                v_pca_data, 
                columns=[f'V_PCA_{i}' for i in range(30)],
                index=df.index
            )
            df = pd.concat([df.drop(columns=v_cols), pca_df], axis=1)

        # 6. Scaling (CRITICAL for GNN Convergence)
        # Scale everything except the target and ID columns
        cols_to_scale = df.select_dtypes(include=[np.number]).columns.difference(['isFraud', 'TransactionID', 'TransactionDT'])
        df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        
        return df