# Advanced Deep Learning-Based DDoS Attack Detection and Classification

This project implements a **deep learning-based framework** for detecting and classifying Distributed Denial of Service (DDoS) attacks using two models:

- **Artificial Neural Network (ANN)** for **binary classification** (benign vs malicious traffic)
- **Graph Neural Network-Multilayer Perceptron (GNN-MLP)** for **multi-class classification** (specific types of DDoS attacks)

The system is built on the **CICDDoS2019** dataset and aims to achieve **high accuracy**, **low computational overhead**, and **real-time detection**.

---

## ✨ Project Highlights

- Binary classification with ANN achieving up to **99.87% accuracy**.
- Multi-class classification with GNN-MLP achieving up to **98.90% accuracy**.
- Comparison between models using 16 and 40 feature sets.
- Use of PyTorch and PyTorch Geometric libraries.
- Feature selection using Decision Trees and grid search optimization.

---

## 📈 Methodology Overview

This study proposes a robust and efficient system for detecting and categorizing Distributed Denial of Service (DDoS) attacks using two models:

- **Artificial Neural Network (ANN)** for binary classification.
- **Graph Neural Network-Multilayer Perceptron (GNN-MLP)** for multi-class classification.

The models are trained and evaluated on the **CICDDoS2019** dataset, covering a wide range of DDoS attack types.

---

### 🔹 Dataset

- **Dataset Used**: [CICDDoS2019 Dataset](https://www.unb.ca/cic/datasets/ddos-2019.html)
- Contains various DDoS attacks like SYN Flood, UDP Flood, MSSQL, LDAP, NetBIOS, and benign traffic.
- Features extracted include flow duration, packet size, byte rate, TCP flag counts, and more.

---

### 🔹 Data Preprocessing

- **Duplicate removal**: Cleaned data to eliminate redundant records.
- **Handling missing values**: Ensured clean and consistent datasets.
- **Feature selection**: 
  - Selected 40 important features using Decision Tree feature importance.
  - Compared model performance with a reduced set of 16 features.
- **Feature scaling**: StandardScaler normalization applied.
- **Label Encoding**: Converted categorical labels to numerical values.
- **Data splitting**:
  - For multi-class classification: 80% train, 10% validation, 10% test.
  - For binary classification: 80% train, 20% test.

---

### 🔹 Model Architectures

- **Artificial Neural Network (ANN)**:
  - Input layer → 1–3 hidden layers (64 neurons each, ReLU activation, dropout 0.3) → Output layer (Sigmoid activation).
  - Optimizer: Adam
  - Loss: Binary Crossentropy
  - Epochs: 50
  - Batch size: 32

- **Graph Neural Network - Multilayer Perceptron (GNN-MLP)**:
  - 2 Graph Convolutional layers + MLP block for classification.
  - Optimizer: Adam
  - Loss: CrossEntropyLoss
  - Learning rate: 0.01
  - Weight decay: 5e-4
  - Epochs: 100
  - Hidden units: 64 channels in GCN layers, 32 units in MLP.

---

### 🔹 Experimental Setup

- Framework: **PyTorch** and **PyTorch Geometric**
- Hardware: 
  - Intel Core i5-11260H CPU
  - 16GB RAM
  - NVIDIA GeForce RTX 3050 GPU
- Code Environment: Jupyter Notebook

---

### 🔹 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **ROC Curve and AUC**

---

## 📊 Results Summary

The performance of the proposed models was evaluated on the CICDDoS2019 dataset, using both 40 and 16 feature sets.

---

### 🔹 Multi-Class Classification (GNN-MLP)

| Metric                  | 40 Features | 16 Features |
|--------------------------|-------------|-------------|
| Accuracy                 | 98.90%      | 98.26%      |
| Macro Avg Precision      | 0.95        | 0.95        |
| Macro Avg Recall         | 0.96        | 0.95        |
| Macro Avg F1-Score       | 0.95        | 0.95        |
| Weighted Avg Precision   | 0.99        | 0.97        |
| Weighted Avg Recall      | 0.99        | 0.97        |
| Weighted Avg F1-Score    | 0.99        | 0.98        |

- **Observation**:
  - The model using 40 features consistently outperformed the 16-feature model.
  - Higher precision and recall observed for critical attack types like SYN and UDP floods.

---

### 🔹 Binary Classification (ANN)

| Metric                  | Value        |
|--------------------------|--------------|
| Accuracy                 | 99.87%       |
| AUC Score                | 0.9998       |

- **Observation**:
  - Simple ANN with 1 hidden layer achieved the best binary classification performance.
  - Increasing hidden layers introduced minor drops in accuracy, showing that a simpler model suffices.

---

### 🔹 Confusion Matrix Insights

- **GNN-MLP (40 Features)**:
  - Most classes correctly classified with minimal confusion.
  - Minor misclassifications observed between similar attack types like MSSQL and UDP.

- **ANN (Binary Classification)**:
  - Very low false positives and false negatives.
  - Strong separation between benign and attack traffic.

---

### 🔹 ROC and Precision-Recall Curves

- **Multi-Class GNN-MLP**:
  - AUC scores close to 1.0 for most classes.
  - NetBIOS class had slightly lower AUC (~0.87), indicating room for further improvement.

- **Binary ANN**:
  - ROC AUC = **0.9998**, indicating near-perfect binary classification capability.

---

The experimental results validate the effectiveness of using GNN-MLP for multi-class DDoS attack detection and ANN for binary detection, achieving state-of-the-art performance with minimal computational cost.


---

The proposed system successfully captures both relational and feature-based information in network traffic to accurately detect and classify DDoS attacks with minimal computational cost.

