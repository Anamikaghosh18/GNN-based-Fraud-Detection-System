import pandas as pd
from config import *

def load_raw_data():
    train_trans = pd.read_csv(TRAIN_TRANSACTION_PATH)
    train_iden  = pd.read_csv(TRAIN_IDENTITY_PATH)
    
    test_trans  = pd.read_csv(TEST_TRANSACTION_PATH)
    test_iden   = pd.read_csv(TEST_IDENTITY_PATH)

    return train_trans, train_iden, test_trans, test_iden


def merge_data(train_trans, train_iden, test_trans, test_iden):
    train_df = train_trans.merge(train_iden, on="TransactionID", how="left")
    test_df  = test_trans.merge(test_iden, on="TransactionID", how="left")

    return train_df, test_df


def run_ingestion():
    train_trans, train_iden, test_trans, test_iden = load_raw_data()
    
    train_df, test_df = merge_data(
        train_trans, train_iden, test_trans, test_iden
    )
    
    os.makedirs(os.path.dirname(PROCESSED_TRAIN_PATH), exist_ok=True)
    train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test_df.to_csv(PROCESSED_TEST_PATH, index=False)

    return train_df, test_df

    