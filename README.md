# GNN-Based Fraud Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Graph Neural%20Network-GNN-0A66C2?style=for-the-badge&logo=databricks&logoColor=white"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Graph%20Library-PyG-F1C40F?style=for-the-badge"/>
   <img src="https://img.shields.io/badge/Status-Active-2ECC71?style=for-the-badge"/>
 
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/2944093d-c831-4380-9399-6049301ee4b3" width="100%" />
</p>

## Overview

A **Graph Neural Network (GNN)** based system to detect fraudulent transactions by modeling **relationships between entities** instead of treating each transaction independently.

By leveraging graph structures, the model captures **hidden connections and behavioral patterns** that traditional approaches often miss.


## ⚡ What Makes It Interesting

* Moves beyond tabular ML → uses **graph-based learning**
* Captures relationships (cards, emails, users, etc.)
* Focuses on **real-world fraud detection challenges**


## 🐳 Run with Docker

```bash
docker build -f services/dl_pipeline/Dockerfile -t gnn-model .
docker run gnn-model
```

## 🤝 Open to

* Ideas on improving GNN performance
* Discussions around fraud detection systems
* Feedback on system design & scalability


## 💭 Quote

> “First make it work, then make it better.”
