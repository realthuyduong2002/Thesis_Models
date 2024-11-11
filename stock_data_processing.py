import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import logging
import yfinance as yf

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pinecone
pc = Pinecone(api_key="09bb980b-6cef-48c3-9aa5-63f3cbc9885e")
index = pc.Index("financial-data-index")

# Initialize SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("SentenceTransformer 'all-MiniLM-L6-v2' loaded successfully.")

# Folder containing the CSV files
folder_path = r'C:\Users\phamt\Downloads\Thesis_Project\Dataset'

# Folder to save the processed CSV files
output_folder = r'C:\Users\phamt\Downloads\Thesis_Project\Processed_Dataset'

# Define the list of stock symbols
symbols = ['NVDA', 'INTC', 'PLTR', 'TSLA', 'AAPL', 'BBD', 'T', 'SOFI',
           'WBD', 'SNAP', 'NIO', 'BTG', 'F', 'AAL', 'NOK', 'BAC',
           'CCL', 'ORCL', 'AMD', 'PFE', 'KGC', 'MARA', 'SLB', 'NU',
           'MPW', 'MU', 'LCID', 'NCLH', 'RIG', 'AMZN', 'ABEV', 'U',
           'LUMN', 'AGNC', 'VZ', 'WBA', 'WFC', 'RIVN', 'UPST', 'GRAB',
           'CSCO', 'VALE', 'AVGO', 'PBR', 'GOOGL', 'SMMT', 'GOLD',
           'CMG', 'BCS', 'UAA']

def fetch_and_save_stock_data(symbol, start_date='2019-01-01', end_date='2024-01-01'):
    """
    Fetch stock data from Yahoo Finance, save to CSV, and upsert data into Pinecone.
    """
    try:
        # Fetch data from Yahoo Finance
        stock_data = yf.download(symbol, start=start_date, end=end_date)

        # Save data to CSV
        csv_file_path = os.path.join(folder_path, f'{symbol}.csv')
        stock_data.to_csv(csv_file_path)
        logger.info(f"Data for {symbol} saved to {csv_file_path}")

        # Load CSV data
        df = pd.read_csv(csv_file_path)

        # Select necessary columns and drop NaN values
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"{csv_file_path} is missing one or more required columns: {required_columns}")
            return

        df = df[required_columns].dropna()

        # Convert each row into a single string of concatenated data
        text_data = df.apply(
            lambda row: f"Date: {row['Date']}, Open: {row['Open']}, High: {row['High']}, Low: {row['Low']}, Close: {row['Close']}, Adj Close: {row['Adj Close']}, Volume: {row['Volume']}",
            axis=1
        )

        # Generate embeddings for the stock data
        embeddings = embedding_model.encode(text_data.tolist(), convert_to_numpy=True)

        # Prepare data for Pinecone upsert
        vectors = [(f"{symbol}_{i}", embeddings[i].tolist(), {'symbol': symbol}) for i in range(len(embeddings))]

        # Upsert data into Pinecone index
        try:
            response = index.upsert(vectors)
            logger.info(f"Successfully upserted {len(vectors)} vectors for {symbol}.")
            logger.info(f"Pinecone response: {response}")
        except Exception as e:
            logger.error(f"Error upserting data into Pinecone for {symbol}: {e}")

        # Save the processed DataFrame to a new CSV file in the output folder
        processed_file_path = os.path.join(output_folder, f'processed_{symbol}.csv')
        df.to_csv(processed_file_path, index=False)
        logger.info(f"Processed data saved to {processed_file_path}")

    except Exception as e:
        logger.error(f"Error processing data for {symbol}: {e}")

def check_missing_symbols():
    """
    Check for missing stock symbols in the folder.
    """
    csv_files_in_folder = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    csv_symbols_in_folder = [f.split('.')[0] for f in csv_files_in_folder]
    missing_symbols = [symbol for symbol in symbols if symbol not in csv_symbols_in_folder]

    if missing_symbols:
        print(f"Missing stock codes: {missing_symbols}")
    else:
        print("There is enough data for all 50 stocks.")