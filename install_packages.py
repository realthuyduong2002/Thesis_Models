import subprocess

# List of libraries to install
packages = [
    "sentence-transformers",
    "pandas",
    "pinecone-client",
    "yfinance",
    "tf-keras",
    "neuralprophet",
    "prophet",
    "xgboost"
]

# Install each library
for package in packages:
    subprocess.run(["pip", "install", "--user", package])