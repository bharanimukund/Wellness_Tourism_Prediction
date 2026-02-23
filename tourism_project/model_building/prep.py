# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/bkrishnamukund/Wellness-Tourism-Prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# -------------------- Drop Identifier --------------------
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True)

# -------------------- Target Column --------------------
target_col = "ProdTaken"


# Standardize Gender column
df['Gender'] = (
    df['Gender']
    .str.strip()              # remove leading/trailing spaces
    .str.replace('Fe Male', 'Female', regex=False)
)

df['Gender'].value_counts(normalize=True) * 100

# Clean MaritalStatus column
df['MaritalStatus'] = (
    df['MaritalStatus']
    .str.strip()
    .str.replace('Unmarried', 'Single', regex=False)
)

df['MaritalStatus'].value_counts(normalize=True) * 100

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="bkrishnamukund/Wellness-Tourism-Prediction",
        repo_type="dataset",
    )
