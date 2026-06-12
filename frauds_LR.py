import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv


# Load .env using explicit path (avoids location issues)
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# Debug: confirm variables are loading
#print(f"KAGGLE_CONFIG_DIR: {os.getenv('KAGGLE_CONFIG_DIR')}")
#print(f"CREDITCARD_DATA_PATH: {os.getenv('CREDITCARD_DATA_PATH')}")

# Read paths from .env
kaggle_config = os.getenv("KAGGLE_CONFIG_DIR")
data_path = os.getenv("CREDITCARD_DATA_PATH")

# Set kaggle config
os.environ["KAGGLE_CONFIG_DIR"] = kaggle_config

# Load dataset
df = pd.read_csv(f"{data_path}/creditcard.csv")
#print(df.head())
print(f"Total de transações : {len(df):,}")
print(f"Fraudes             : {df['Class'].sum():,}  ({df['Class'].mean()*100:.2f}%)")
#print(df["Class"].value_counts(normalize=True))

df["Amount_Log"] = np.log1p(df["Amount"])
#print(df["Amount_Log"])

x = df.drop("Class", axis=1)
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size= 0.3, random_state= 42
)

scaler = StandardScaler()
x_train = x_train.copy()
x_test = x_test.copy()
x_train[["Amount", "Time"]] = scaler.fit_transform(x_train[["Amount", "Time"]])
x_test[["Amount", "Time"]]  = scaler.transform(x_test[["Amount", "Time"]])

smote = SMOTE()
x_res, y_res = smote.fit_resample(x_train, y_train)

model = LogisticRegression(max_iter=1000)
model.fit(x_res, y_res)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

y_score = model.predict_proba(x_test)[:, 1]  
precision, recall, _ = precision_recall_curve(y_test, y_score)
plt.figure(figsize=(8, 5))
plt.plot(recall, precision, marker=".", linewidth=1.5)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.tight_layout()
plt.show()