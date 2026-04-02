import os


current_file_path = os.path.abspath(__file__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))

# Define the Data Directory using the Root
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Define Raw Paths (Absolute)
TRAIN_TRANSACTION_PATH = os.path.join(DATA_DIR, "raw", "train_transaction.csv")
TRAIN_IDENTITY_PATH    = os.path.join(DATA_DIR, "raw", "train_identity.csv")
TEST_TRANSACTION_PATH  = os.path.join(DATA_DIR, "raw", "test_transaction.csv")
TEST_IDENTITY_PATH     = os.path.join(DATA_DIR, "raw", "test_identity.csv")

# Define Processed Paths (Absolute)
PROCESSED_TRAIN_PATH = os.path.join(DATA_DIR, "processed", "train.csv")
PROCESSED_TEST_PATH  = os.path.join(DATA_DIR, "processed", "test.csv")

