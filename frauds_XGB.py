import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size= 0.3, random_state= 42
)
xgb.fit(x_train, y_train)
y_pred_xgb = xgb.predict(x_test)
print(classification_report(y_test, y_pred_xgb))
