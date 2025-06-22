# Drug - Target Interaction (DTI) Prediction Using FDA-Approved BRCA Chemotherapy Data

AUTHOR: Vitika Patel
Tech Stack: Python, Jupyter, RDKit, XGBoost, SHAP, scikit-learn, NumPy, pandas


## Project Overview

This project builds a machine learning pipeline to predict drug–target interactions (DTIs) for FDA-approved breast cancer chemotherapy agents, using real-world biochemical binding data from BindingDB.

By extracting molecular and sequence-level features and training multiple classifiers, the model accurately predicts whether a given drug–protein pair interacts — a critical step in drug discovery and drug repurposing.

## Dataset

SOURCE: https://www.kaggle.com/datasets/aliabedimadiseh/fda-approved-brca-chemotherapy-drugs-in-bindingdb
FORMAT: Combined .csv and .tsv files

## CONTENTS

Drug info: SMILES string and Ligands IDs
Protein info: FASTA sequences, UniProt IDs
Binding data: Ki, IC50, Kd values in nM

## OBJECTIVE

Predict whether a drug–protein pair shows binding interaction (active) or not (inactive), based on structural and sequence-level features.

## METHODOLOGY

1. Data Cleaning and Integration
   * Merged 8+ files into a single DTI dataframe
   * Removed null/incomplete rows
   * Labeled interactions as:
     a. Active (1) if Ki/Kd/IC50 < 1000 nM
     b. Inactive (0) otherwise
2. Feature Engineering
   * Drugs: Generated ECFP4 fingerprints (1024-bit) using RDKit from SMILES strings
   * Proteins: Computed Amino Acid Composition (AAC) — frequency of each of the 20 standard amino acids
3. Model Training
   * Combined drug and protein features → final feature vector of size 1044
   * Trained and evaluated:
     a. Random Forest
     b. XGBoost
4. Model Evaluation
   * Metrics: Accuracy, ROC AUC, Confusion Matrix
   * Results:
     a. AUC = 1.0
     b. XGBoost: AUC = 0.99 (strong generalization)
5. Model Interpretation (SHAP)
   * Used SHAP to explain feature contributions
   * Found key fingerprint bits and amino acid compositions that drove predictions
   * Demonstrated that the model learned meaningful biochemical relationships
  
## RESULTS

| Model         | Accuracy | ROC AUC |
| ------------- | -------- | ------- |
| Random Forest | 1.00     | 1.00    |
| XGBoost       | 0.97     | 0.99    |

* Top features included specific ECFP bits (structural motifs) and amino acid compositions (e.g., leucine frequency)
* SHAP summary plots confirmed interpretable patterns in predictions

## TOOLS AND LIBRARIES

* RDKit for molecular feature extraction (SMILES → ECFP)
* Biopython + pandas for protein sequence handling
* scikit-learn for training + metrics
* XGBoost for advanced modeling
* SHAP for model interpretability

## KEY TAKEAWAYS

* Real-world biochemical data can be modeled with basic structural + sequence features
* XGBoost + SHAP gives performance and transparency
* This pipeline could be extended to other diseases, drugs, or protein targets

## PROJECT STRUCTURE

DTI-Predictor
* data                      # Raw and cleaned BindingDB files
* dti_pipeline.ipynb         # Main Jupyter notebook
* model                    # Trained models (optional .pkl/.joblib)
* figures                  # ROC curves, SHAP plots
* README.md                  # Project summary

## HOW TO RUN

1. Clone the repo.
2. Install dependencies:
     pip install -r requirements.txt
3. Open dti_pipeline.ipynb in Jupyter Lab
4. Run all cells to train, evaluate, and interpret the model

## FUTURE WORK

* Add 3D protein structure features (e.g. from AlphaFold or PDB)
* Test generalization across other cancer types or datasets
* Deploy as a web app or API
