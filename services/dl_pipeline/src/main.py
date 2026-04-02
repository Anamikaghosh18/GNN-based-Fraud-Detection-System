import os
import torch
import pandas as pd 
from data_pipeline.ingest import run_ingestion
from data_pipeline.preprocess import Preprocessor
from graph_builder.graph_builder import GraphBuilder
from config import *

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
        print(f"✅ Saved cleaned data to: {FINAL_CLEANED_PATH}")
    else:
        print(f"Loading existing cleaned data from: {FINAL_CLEANED_PATH}")
        cleaned_df = pd.read_csv(FINAL_CLEANED_PATH)
    

    if not os.path.exists(GRAPH_OBJ_PATH):
        print("🕸️ Constructing Graph from cleaned data...")
        
        # 1. Load the cleaned dataframe (if not already in memory)
        df = pd.read_csv(FINAL_CLEANED_PATH)
        
        # 2. Build the Graph
        builder = GraphBuilder()
        graph_data = builder.construct_hetero_graph(df)
        
        # 3. Save the PyTorch Object 
        torch.save(graph_data, GRAPH_OBJ_PATH)
        print(f"✅ Graph object saved to {GRAPH_OBJ_PATH}")
    else:
        print(f"Graph object already exists at {GRAPH_OBJ_PATH}")
        graph_data = torch.load(GRAPH_OBJ_PATH)

    print(f"Ready for training!")
if __name__ == "__main__":
    main()