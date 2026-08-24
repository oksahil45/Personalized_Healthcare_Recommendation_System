
import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Cleaned_Dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.joblib")

FEATURES = [
    "fever", "cough", "fatigue", "difficulty_breathing",
    "age", "gender", "blood_pressure", "cholesterol_level"
]

def main():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().dropna(subset=["disease"]).copy()

    X = df[FEATURES]
    y = df["disease"].astype(str)

    categorical_features = [
        "fever", "cough", "fatigue", "difficulty_breathing", "gender"
    ]
    numeric_features = ["age", "blood_pressure", "cholesterol_level"]

    preprocessor = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), numeric_features),
    ])

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced",
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Training rows: {len(df)}")
    print(f"Disease classes: {y.nunique()}")

if __name__ == "__main__":
    main()
