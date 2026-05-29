# Network Intrusion Detection System
I built this project to detect network intrusions and classify malicious traffic using machine learning. The model is trained on the CICIDS2017 dataset containing 2.15 million real network flow records across 15 different attack categories.

The biggest challenge was handling the massive class imbalance in the dataset — I used SMOTE to fix this and applied correlation-based feature selection to reduce 78 features down to the top 30, which actually improved the model's performance. Random Forest achieved 99.89% accuracy and XGBoost achieved 99.86%.

I also built a Flask REST API on top of the model so it can predict in real-time for single records as well as process bulk CSV files with results export.

## Project Files
- `app.py` — Flask REST API for real-time and batch prediction
- `eda_cleaning.ipynb` — Data cleaning and exploratory data analysis
- `traing.ipynb` — Model training, feature selection, and SMOTE
- `model.ipynb` — Dataset loading and initial data cleaning

## Model Performance
| Model | Accuracy |
|-------|----------|
| Random Forest | **99.89%** |
| XGBoost | **99.86%** |
| Logistic Regression | Baseline |

## Key Features
- Multi-class classification across 15 network attack types
- Flask REST API for real-time single-record prediction
- Bulk CSV batch processing with downloadable results export
- Correlation-based feature selection (top 30 of 78 features)
- Class imbalance handling using SMOTE
- End-to-end EDA and feature engineering pipeline

## Tech Stack
| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| ML Models | Random Forest, XGBoost, Logistic Regression |
| ML Libraries | Scikit-learn, XGBoost, imbalanced-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web & API | Flask, REST API |
| Model Saving | Joblib |
| Environment | Jupyter Notebook, VS Code |

## Dataset
**CICIDS2017** — Canadian Institute for Cybersecurity  
2.15M records | 78 features | 15 attack categories  
[Dataset Link](https://www.unb.ca/cic/datasets/ids-2017.html)
