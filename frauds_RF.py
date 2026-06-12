import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv


# Load .env using explicit path (avoids location issues)
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)


# Read paths from .env
kaggle_config = os.getenv("KAGGLE_CONFIG_DIR")
data_path = os.getenv("CREDITCARD_DATA_PATH")
# Set kaggle config
os.environ["KAGGLE_CONFIG_DIR"] = kaggle_config

# Load dataset
df = pd.read_csv(f"{data_path}/creditcard.csv")
print(f"Total de transações : {len(df):,}")
print(f"Fraudes : {df['Class'].sum():,}  ({df['Class'].mean()*100:.2f}%)")
df["Class"].value_counts(normalize=True)

df["Amount_Log"] = np.log1p(df["Amount"])
df = df.drop(columns=["Amount"])

x = df.drop("Class", axis=1)
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size= 0.3, random_state= 42
)

scaler = StandardScaler()
x_train = x_train.copy()
x_test = x_test.copy()
x_train[["Amount_Log", "Time"]] = scaler.fit_transform(x_train[["Amount_Log", "Time"]])
x_test[["Amount_Log", "Time"]]  = scaler.transform(x_test[["Amount_Log", "Time"]])

smote = SMOTE()
x_res, y_res = smote.fit_resample(x_train, y_train)

rf = RandomForestClassifier(
    n_estimators=250,
    max_depth=10,
    n_jobs= 1,
    random_state= 42
)
rf.fit(x_res, y_res)
y_score = rf.predict_proba(x_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_score)
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
best_thr = thresholds[np.argmax(f1_scores[:-1])]
y_pred_rf = (y_score >= best_thr).astype(int)
print(classification_report(y_test, y_pred_rf))

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — Random Forest")
plt.show()