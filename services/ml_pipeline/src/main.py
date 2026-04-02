import os
import pandas as pd 
from data_pipeline.ingest import run_ingestion
from data_pipeline.preprocess import Preprocessor

# --- PATH LOGIC ---
current_file_path = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FINAL_CLEANED_PATH = os.path.join(PROCESSED_DIR, "cleaned_data.csv")

def main():
    
    if not os.path.exists(FINAL_CLEANED_PATH):
        print("Starting Ingestion...")
        train_df, test_df = run_ingestion() 
        
        print("Starting Preprocessing...")
        processor = Preprocessor()
        cleaned_df = processor.transform(train_df)
        
        
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        #  Save using the ABSOLUTE path
        cleaned_df.to_csv(FINAL_CLEANED_PATH, index=False)
        print(f"Saved cleaned data to: {FINAL_CLEANED_PATH}")
    else:
        print(f"Loading existing cleaned data from: {FINAL_CLEANED_PATH}")
        cleaned_df = pd.read_csv(FINAL_CLEANED_PATH)

if __name__ == "__main__":
    main()