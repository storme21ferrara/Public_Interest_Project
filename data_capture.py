import os
import logging
import pandas as pd
import json
import xml.etree.ElementTree as ET
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from bs4 import BeautifulSoup
import pyarrow.parquet as pq
import fastavro
import requests
import fitz  # PyMuPDF
import docx
from email import policy
from email.parser import BytesParser
import spacy

# Mocking some dependencies and functions that cannot be tested in this environment
def fitz_open_mock(file_path):
    class MockDoc:
        def get_text(self):
            return "Mock text from PDF page"
    return MockDoc()

def docx_Document_mock(file_path):
    class MockDoc:
        paragraphs = ["Mock text from DOCX paragraph"]
    return MockDoc()

fitz.open = fitz_open_mock
docx.Document = docx_Document_mock

# Setup logging
def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=log_path, filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load configuration
def load_config(config_path):
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        return config
    except FileNotFoundError as e:
        logging.error(f"Config file not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from config file: {e}")
        raise

# Validation function for file existence
def validate_file(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return False
    return True

# Generic data capture function for various file types
def capture_data(file_path, file_type):
    if not validate_file(file_path):
        return None
    try:
        if file_type == 'csv':
            data = pd.read_csv(file_path)
        elif file_type == 'excel':
            data = pd.read_excel(file_path)
        elif file_type == 'json':
            data = pd.read_json(file_path)
        elif file_type == 'xml':
            tree = ET.parse(file_path)
            root = tree.getroot()
            data = [{child.tag: child.text for child in elem} for elem in root]
            data = pd.DataFrame(data)
        elif file_type == 'html':
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                data = [[col.text for col in row.find_all('td')] for row in soup.find_all('tr')]
                data = pd.DataFrame(data)
        elif file_type == 'txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                data = file.readlines()
            data = pd.DataFrame(data, columns=["text"])
        elif file_type == 'parquet':
            data = pq.read_table(file_path).to_pandas()
        elif file_type == 'avro':
            with open(file_path, 'rb') as f:
                reader = fastavro.reader(f)
                data = [record for record in reader]
            data = pd.DataFrame(data)
        elif file_type == 'pdf':
            doc = fitz.open(file_path)
            text = "".join(page.get_text() for page in doc)
            data = pd.DataFrame([text], columns=["Content"])
        elif file_type == 'docx':
            doc = docx.Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs)
            data = pd.DataFrame([text], columns=["Content"])
        elif file_type == 'eml':
            with open(file_path, 'rb') as file:
                msg = BytesParser(policy=policy.default).parse(file)
            text = msg.get_body(preferencelist=('plain')).get_content()
            data = pd.DataFrame([text], columns=["Content"])
        elif file_type == 'url':
            response = requests.get(file_path)
            response.raise_for_status()
            data = response.json()
            data = pd.DataFrame(data)
        else:
            logging.warning(f"Unsupported file type: {file_type}")
            return None
        logging.info(f"Data captured from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error capturing data from {file_path}: {e}")
        return None

# Function to dynamically add columns if needed
def add_columns_if_needed(df, required_columns):
    for column in required_columns:
        if column not in df.columns:
            df[column] = None
            logging.info(f"Added missing column: {column}")
    return df

# Function to capture data from a folder
def capture_data_from_folder(folder_path, db_path, table_name):
    try:
        engine = create_engine(f'sqlite:///{db_path}')
        Session = sessionmaker(bind=engine)
        session = Session()
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = file.split('.')[-1].lower()
                file_type = {
                    'csv': 'csv',
                    'xlsx': 'excel',
                    'json': 'json',
                    'xml': 'xml',
                    'html': 'html',
                    'txt': 'txt',
                    'parquet': 'parquet',
                    'avro': 'avro',
                    'pdf': 'pdf',
                    'docx': 'docx',
                    'eml': 'eml',
                }.get(file_ext)
                if file_type:
                    data = capture_data(file_path, file_type)
                    if data is not None:
                        # Add columns dynamically if needed
                        data = add_columns_if_needed(data, ['additional_column_1', 'additional_column_2'])
                        data.to_sql(table_name, engine, if_exists='append', index=False)
                        logging.info(f"Data from {file_path} captured and saved to database.")
                else:
                    logging.warning(f"Unsupported file type: {file_path}")
        session.commit()
        session.close()
        logging.info(f"Data capture from folder {folder_path} completed successfully.")
    except Exception as e:
        logging.error(f"Error capturing data from folder {folder_path}: {e}")

# Main execution with logging and configuration
if __name__ == "__main__":
    config_path = "E:/Public_Interest_Project/config_files/terminal_config.json"
    config = load_config(config_path)
    setup_logging(config['subdirectories']['logs'] + 'data_capture.log')
    capture_data_from_folder(config['input_data_location'], config['output_data_location'] + 'metadata.db', 'captured_data')
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
