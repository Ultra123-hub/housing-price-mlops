import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.read_csv("data/nigeria_housing_data_v1.csv")

X = data.drop("price_ngn", axis=1)
y = data["price_ngn"]

numeric_features = [
    "area_sqm",
    "bedrooms",
    "bathrooms",
    "parking_spaces",
    "distance_to_city_center_km",
    "electricity_hours_per_day"
]

categorical_features = ["city", "property_type"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save the splits and preprocessor
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

# Save preprocessor
with open("data/preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

print("Preprocessing complete. Data saved.")