import os

current_file_path = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Define Raw Paths (Absolute)
TRAIN_TRANSACTION_PATH = os.path.join(DATA_DIR, "raw", "train_transaction.csv")
TRAIN_IDENTITY_PATH    = os.path.join(DATA_DIR, "raw", "train_identity.csv")
TEST_TRANSACTION_PATH  = os.path.join(DATA_DIR, "raw", "test_transaction.csv")
TEST_IDENTITY_PATH     = os.path.join(DATA_DIR, "raw", "test_identity.csv")

# --- PATH LOGIC ---

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
PROCESSED_TEST_PATH  = os.path.join(PROCESSED_DIR, "test.csv")
FINAL_CLEANED_PATH = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
GRAPH_OBJ_PATH = os.path.join(DATA_DIR, "processed", "graph_data.pt")
