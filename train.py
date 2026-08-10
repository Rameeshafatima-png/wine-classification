import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "wine_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "model_bundle.pkl"


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully")
print("-" * 40)

print(df.head())

print("\nDataset Shape:")
print(df.shape)


# ==========================================
# SELECT ONLY 3 FEATURES
# ==========================================

FEATURES = [
    "alcohol",
    "malic_acid",
    "proline"
]

TARGET = "target"


# ==========================================
# CHECK COLUMNS
# ==========================================

missing_columns = [
    column
    for column in FEATURES + [TARGET]
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"These columns were not found: {missing_columns}"
    )


# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df[FEATURES]

y = df[TARGET]


print("\nSelected Features:")
print(FEATURES)

print("\nTarget:")
print(TARGET)


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ==========================================
# PREDICTION
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


# ==========================================
# CLASS NAMES
# ==========================================

target_names = [
    "Class 0",
    "Class 1",
    "Class 2"
]


# ==========================================
# SAVE MODEL
# ==========================================

model_bundle = {

    "model": model,

    "features": FEATURES,

    "target_names": target_names,

    "accuracy": accuracy,

    "version": "2.0.0"
}


joblib.dump(
    model_bundle,
    MODEL_PATH
)


print("\nModel saved successfully!")

print(
    f"Location: {MODEL_PATH}"
)

print("\nFeatures used by model:")

for feature in FEATURES:
    print(f"- {feature}")