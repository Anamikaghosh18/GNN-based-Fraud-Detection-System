import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


class Preprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=30)

    def transform(self, df):
        # 1. Drop high-null columns (>90%)
        null_pct = df.isnull().sum() / len(df)
        df = df.drop(columns=null_pct[null_pct > 0.90].index)

        # 2. Fill NaNs (numeric)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(-999)

        df['uid'] = (
            df['card1'].astype(str) + '_' +
            df['card2'].astype(str) + '_' +
            df['addr1'].astype(str) + '_' +
            df['P_emaildomain'].astype(str)
        )


        # 3. Time features
        df['hour'] = (df['TransactionDT'] // 3600) % 24
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # 4. Encode ALL categorical safely
        for col in df.columns:
            if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
                if col == 'uid':  # skip if used for graph edges
                    continue
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # 5. PCA for V columns
        v_cols = [c for c in df.columns if c.startswith('V')]
        if v_cols:
            v_pca = self.pca.fit_transform(df[v_cols])

            pca_df = pd.DataFrame(
                v_pca,
                columns=[f'V_PCA_{i}' for i in range(v_pca.shape[1])],
                index=df.index
            )

            df = pd.concat([df.drop(columns=v_cols), pca_df], axis=1)

        # 6. Scale numeric features
        cols_to_scale = df.select_dtypes(include=[np.number]).columns.difference(
            ['isFraud', 'TransactionID', 'TransactionDT']
        )

        df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])

        
        uid_col = df['uid']  # save
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(0)
        df['uid'] = uid_col  # restore


        return df