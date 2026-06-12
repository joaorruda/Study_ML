import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv


dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# Read paths from .env
kaggle_config = os.getenv("KAGGLE_CONFIG_DIR")
data_path = os.getenv("CREDITCARD_DATA_PATH")
os.environ["KAGGLE_CONFIG_DIR"] = kaggle_config

# Load dataset
df = pd.read_csv(f"{data_path}/creditcard.csv")
print(f"Total de transações : {len(df):,}")
print(f"Fraudes             : {df['Class'].sum():,}  ({df['Class'].mean()*100:.2f}%)")
df["Class"].value_counts(normalize=True)
df["Amount_Log"] = np.log1p(df["Amount"])

x = df.drop("Class", axis=1)
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size= 0.3, random_state= 42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])
x_train = x_train.copy()
x_test = x_test.copy()
x_train[["Amount", "Time"]] = pipeline.named_steps['scaler'].fit_transform(x_train[["Amount", "Time"]])
x_test[["Amount", "Time"]]  = pipeline.named_steps['scaler'].transform(x_test[["Amount", "Time"]])

smote = SMOTE()
x_res, y_res = smote.fit_resample(x_train, y_train)
pipeline.fit(x_res, y_res)

y_pred = pipeline.predict(x_test)
threshold = 0.35
y_pred_custom = (y_pred >= threshold).astype(int)
print(classification_report(y_test, y_pred_custom))

y_score = pipeline.predict_proba(x_test)[:, 1]  
precision, recall, _ = precision_recall_curve(y_test, y_score)
