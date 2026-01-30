# QSAR Workflow (RDKit + Deep Learning)

A reproducible **QSAR modeling workflow** combining **RDKit molecular descriptors**, **MACCS fingerprints**, **outlier detection**, **applicability domain analysis**, and a **Keras-based neural network** for regression tasks.

This repository contains a **cleaned and fixed version** of an original QSAR script, with improved stability, validation, and error handling.

---

## 📌 Overview

**Author:** Divya Karade  
**Model type:** Regression (Neural Network)  
**Use case:** Binding affinity / docking score prediction  
**Descriptors:** RDKit 2D descriptors + MACCS keys  
**Outlier detection:** PCA + Isolation Forest  
**Applicability Domain (AD):** Descriptor range-based check  
**Frameworks:** RDKit, scikit-learn, TensorFlow / Keras  

---

## 🧪 Workflow Steps

1. **Data Loading**
   - Reads molecular data from a CSV file (`SpikeRBD_DD.csv`)
   - Requires SMILES strings and a numeric target value

2. **Descriptor Generation**
   - RDKit 2D molecular descriptors
   - MACCS fingerprints (167 bits)

3. **Feature Cleaning**
   - Removes NaN and infinite values
   - Clips extreme descriptor values
   - Ensures numeric consistency

4. **Train/Test Split**
   - 70/30 split with fixed random seed
   - Standard scaling applied to features

5. **Outlier Detection**
   - PCA (2 components) for dimensionality reduction
   - Isolation Forest to remove anomalous samples

6. **Applicability Domain (AD)**
   - Checks whether input molecule descriptors fall within
     the min–max range of the training set

7. **Model Training**
   - Fully connected neural network (Keras)
   - Early stopping based on validation R²
   - Custom metrics: RMSE and R²

8. **Evaluation**
   - MAE, MSE, RMSE, and R² on training and test sets

9. **Prediction**
   - Predicts binding affinity for new SMILES
   - Outputs applicability domain status

---

## 📂 Input Data Format

### Training Dataset (`SpikeRBD_DD.csv`)

Required columns:
- `smiles` — SMILES representation of molecules
- `DockingScore` — Target regression value

Example:
```csv
smiles,DockingScore
CCO,-6.5
CCN,-7.1

## 🧬 Predicting New Molecules

New molecules can be provided as a Python list:

```python
smiles = ["CCO"]  # Example: ethanol

The script performs the following steps:

- Validates SMILES strings
- Generates RDKit 2D descriptors and MACCS fingerprints
- Checks applicability domain
- Predicts binding affinity using the trained neural network

## 📈 Model Architecture
Input Layer:  Descriptor dimension
Dense Layer:  600 neurons (ReLU)
Dense Layer:  100 neurons (ReLU)
Dense Layer:  100 neurons (ReLU)
Output Layer: 1 neuron (Linear)

### Training configuration

- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Batch size:** 400
- **Epochs:** Up to 200 (early stopping enabled)

---

## 📊 Evaluation Metrics

The following metrics are reported for both **training** and **test** sets:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

---

## 🛠 Requirements

- Python ≥ 3.8
- RDKit
- TensorFlow / Keras
- scikit-learn
- pandas
- numpy
- matplotlib

## Installation (example)
pip install numpy pandas scikit-learn tensorflow matplotlib
conda install -c conda-forge rdkit

## ⚠️ Notes & Limitations

- Applicability domain is based on **descriptor range checks**, not distance-based confidence metrics.
- Isolation Forest contamination rate is fixed at **10%**.
- Designed for **QSAR regression**, not classification.
- Best used with **chemically consistent datasets**.

---

## 📜 License

This code is provided for **research and educational purposes**.  
Please cite appropriately if used in academic or industrial work.

---

## 👩‍🔬 Author

**Divya Karade**  
Cheminformatician | QSAR | AI/ML
