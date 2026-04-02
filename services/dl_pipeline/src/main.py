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
        
        builder = GraphBuilder()
        graph_data = builder.construct_hetero_graph(df)

        # 1. num_nodes here for a new graph
        num_nodes = graph_data['transaction'].num_nodes
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[:int(num_nodes * 0.8)] = True
        val_mask = ~train_mask
        
        graph_data['transaction'].train_mask = train_mask
        graph_data['transaction'].val_mask = val_mask

        torch.save(graph_data, GRAPH_OBJ_PATH)
        print(f"✅ Graph object saved to {GRAPH_OBJ_PATH}")
    else:
        print(f"Graph object already exists at {GRAPH_OBJ_PATH}")
        graph_data = torch.load(GRAPH_OBJ_PATH)
        
        # 2. num_nodes here for an existing graph
        num_nodes = graph_data['transaction'].num_nodes

    
    print("\n" + "="*30)
    print("🎉 PIPELINE COMPLETE")
    print(f"Graph Nodes: {num_nodes}")
    print(f"Fraud Ratio: {float(graph_data['transaction'].y.sum() / num_nodes):.2%}")
    print("="*30)

if __name__ == "__main__":
    main()