import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_TRANSACTION_PATH = "data/raw/train_transaction.csv"
TRAIN_IDENTITY_PATH    = "data/raw/train_identity.csv"
TEST_TRANSACTION_PATH  = "data/raw/test_transaction.csv"
TEST_IDENTITY_PATH     = "data/raw/test_identity.csv"

PROCESSED_TRAIN_PATH = "data/processed/train.csv"
PROCESSED_TEST_PATH = "data/processed/test.csv"