# MediScope — Personalized Healthcare & Medicine Recommendation System

MediScope is a portfolio-focused machine learning and Streamlit application for exploring a supplied healthcare dataset, predicting disease categories from structured patient attributes, and displaying reference-based care information.

> **Important:** This is an educational/portfolio prototype. It is not a clinically validated diagnostic device and does not provide prescriptions.

## Features

- Interactive patient assessment
- Random Forest disease-category prediction
- Top-5 probability ranking
- Healthcare analytics with Plotly
- Symptom and risk-level exploration
- Data-quality checks
- Reference-based medicine/care information from the supplied project database
- Model configuration and transparency page
- GitHub + Streamlit Cloud ready

## Project structure

```text
Personalized-Healthcare-Recommendation-System/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── Cleaned_Dataset.csv
│   └── medicine_database.pkl
└── models/
    └── disease_model.joblib
```

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The repository already contains a trained model, so `python train_model.py` is only needed if you modify the dataset or want to rebuild the model.

## Dataset notes

The supplied CSV contains 349 rows and 14 columns. It contains 49 exact duplicate rows and 116 disease categories. The model-training script removes exact duplicates before training.

Because many disease categories have very few observations, conventional holdout accuracy can be unstable and is intentionally not presented as a clinical-quality metric.

## Streamlit deployment

1. Upload the complete repository to GitHub.
2. In Streamlit Community Cloud, create a new app.
3. Select the GitHub repository.
4. Branch: `main`
5. Main file: `app.py`
6. Deploy.

## Technology

Python, Pandas, NumPy, Scikit-learn, Random Forest, Plotly, Streamlit, Joblib.
