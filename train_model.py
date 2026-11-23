# train_model.py:  trains an F1 podium predictor and saves it
# run once: python train_model.py
# creates model.joblib in the same folder
# Soumay Patidar | DATS 6401

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

data_dir = Path(__file__).resolve().parent / "data"

# load and merge
results = pd.read_csv(data_dir / "results.csv", na_values="\\N")
races = pd.read_csv(data_dir / "races.csv", na_values="\\N")
constructors = pd.read_csv(data_dir / "constructors.csv", na_values="\\N")

df = results.merge(races[["raceId", "year", "round", "circuitId"]], on="raceId")
df = df.merge(constructors[["constructorId", "constructorRef"]], on="constructorId")

# use modern era only (2000+), more relevant to current F1
df = df[df["year"] >= 2000].copy()

# target: did they finish on the podium (top 3)?
df["podium"] = (df["positionOrder"] <= 3).astype(int)

#  feature engineering

# 1. grid position (0 = pit lane start, treat as worst)
df["grid"] = df["grid"].fillna(20)
df.loc[df["grid"] == 0, "grid"] = 20  # pit lane start

# 2. constructor historical podium rate (how strong is the team?)
constructor_stats = df.groupby("constructorRef")["podium"].mean()
df["constructor_podium_rate"] = df["constructorRef"].map(constructor_stats)

# 3. encode constructor as a number for the model
le_constructor = LabelEncoder()
df["constructor_encoded"] = le_constructor.fit_transform(df["constructorRef"])

# 4. number of laps (longer races = more chance for things to go wrong)
df["laps"] = df["laps"].fillna(0)

# 5. circuit (some tracks favor certain grid positions more)
le_circuit = LabelEncoder()
df["circuit_encoded"] = le_circuit.fit_transform(df["circuitId"].astype(str))

# features we'll use
feature_cols = [
    "grid",
    "constructor_podium_rate",
    "constructor_encoded",
    "laps",
    "circuit_encoded",
    "round",
]

# drop any rows with missing values in our features
df_clean = df.dropna(subset=feature_cols + ["podium"])

X = df_clean[feature_cols]
y = df_clean["podium"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"Test accuracy: {acc:.2%}")

# also check how well it does on podium predictions specifically
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["No podium", "Podium"]))

# save everything the server needs
bundle = {
    "model": model,
    "feature_names": feature_cols,
    "le_constructor": le_constructor,
    "le_circuit": le_circuit,
    "constructor_stats": constructor_stats.to_dict(),
    "known_constructors": list(le_constructor.classes_),
    "known_circuits": list(le_circuit.classes_),
}
joblib.dump(bundle, Path(__file__).resolve().parent / "model.joblib")
print("\nSaved model.joblib")
print(f"Known constructors: {list(le_constructor.classes_)[:10]}...")
