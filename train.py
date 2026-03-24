import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from kneed import KneeLocator

print("Loading dataset...")

df = pd.read_csv("smartcart_customers.csv")

# ================= FEATURE ENGINEERING =================

df["Income"] = df["Income"].fillna(df["Income"].median())
df["Age"] = 2026 - df["Year_Birth"]

df["Total_Spending"] = (
    df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
    + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
)

df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

df["Education"] = df["Education"].replace({
    "Basic": "Undergraduate",
    "2n Cycle": "Undergraduate",
    "Graduation": "Graduate",
    "Master": "Postgraduate",
    "PhD": "Postgraduate"
})

df["Living_With"] = df["Marital_Status"].replace({
    "Married": "Partner",
    "Together": "Partner",
    "Single": "Alone",
    "Divorced": "Alone",
    "Widow": "Alone",
    "Absurd": "Alone",
    "YOLO": "Alone"
})

df = df[df["Age"] < 90]
df = df[df["Income"] < 600000]

drop_cols = [
    "ID","Year_Birth","Marital_Status",
    "Kidhome","Teenhome","Dt_Customer",
    "MntWines","MntFruits","MntMeatProducts",
    "MntFishProducts","MntSweetProducts","MntGoldProds"
]

df = df.drop(columns=drop_cols)

# ================= ENCODING =================

ohe = OneHotEncoder()
cat_cols = ["Education","Living_With"]

enc = ohe.fit_transform(df[cat_cols]).toarray()
enc_df = pd.DataFrame(enc, columns=ohe.get_feature_names_out())

df = pd.concat([df.drop(columns=cat_cols), enc_df], axis=1)

df = df.select_dtypes(include=np.number)

# ================= PIPELINE =================

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_scaled = pipeline.fit_transform(df)

# ================= PCA =================

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# ================= OPTIMAL K =================

wcss = []
for k in range(1, 10):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_pca)
    wcss.append(model.inertia_)

knee = KneeLocator(range(1,10), wcss, curve="convex", direction="decreasing")
optimal_k = knee.elbow

print("Optimal clusters:", optimal_k)

# ================= FINAL MODEL =================

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(X_pca)

df["Cluster"] = labels

# ================= SAVE ARTIFACTS =================

joblib.dump(kmeans, "model.pkl")
joblib.dump(pipeline, "scaler_pipeline.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(ohe, "encoder.pkl")

df.to_csv("processed_data.csv", index=False)

print("Training complete")
print("Artifacts saved")