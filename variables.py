import os
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from dotenv import load_dotenv


dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# Read paths from .env
kaggle_config = os.getenv("KAGGLE_CONFIG_DIR")
data_path = os.getenv("CREDITCARD_DATA_PATH")
os.environ["KAGGLE_CONFIG_DIR"] = kaggle_config

# Load dataset
df = pd.read_csv(f"{data_path}/creditcard.csv")

x = df.drop("Class", axis=1)
y = df["Class"]

xgb = XGBClassifier(
    scale_pos_weight = 10,
    use_label_encoder=False,
    eval_metric="logloss"
)

importancia = xgb.feature_importances_
plt.bar(range(len(importancia)), importancia)
plt.show()