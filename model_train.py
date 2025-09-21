import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# ✅ 1. Load Dataset
df = pd.read_csv("D:\PW Data Science\Project\Heart Prediction\heart.csv")  # apna dataset ka path do (kaggle heart disease dataset)

# ✅ 2. Features and Target
X = df.drop("target", axis=1)   # features
y = df["target"]               # target column (0/1)

# ✅ 3. Save expected columns
joblib.dump(X.columns.tolist(), "columns.pkl")

# ✅ 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

# ✅ 6. Train Model (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# ✅ 7. Evaluate
y_pred = knn.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ✅ 8. Save Model
joblib.dump(knn, "KNN_heart.pkl")
