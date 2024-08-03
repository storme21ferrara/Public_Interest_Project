import os
import re
import logging
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import docx
import matplotlib.pyplot as plt
import seaborn as sns
from openvino.inference_engine import IECore
from transformers import pipeline
from textblob import TextBlob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomError(Exception):
    """Base class for custom exceptions"""
    pass

class DataLoadingError(CustomError):
    """Exception raised for errors in the data loading process."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class ModelTrainingError(CustomError):
    """Exception raised for errors during model training."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class EvaluationError(CustomError):
    """Exception raised for errors during model evaluation."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def load_config(config_path='file_management_config.json'):
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        return config
    except FileNotFoundError as e:
        logging.error(f"Config file not found: {e}")
        raise DataLoadingError("Configuration file is missing.")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from config file: {e}")
        raise DataLoadingError("Error decoding JSON configuration.")

def load_data(file_path, file_type='csv'):
    try:
        if file_type == 'csv':
            data = pd.read_csv(file_path)
        elif file_type == 'excel':
            data = pd.read_excel(file_path)
        elif file_type == 'json':
            data = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        return data
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise DataLoadingError("Data file is missing.")
    except ValueError as e:
        logging.error(f"Error loading data: {e}")
        raise DataLoadingError("Error loading data from file.")
    except Exception as e:
        logging.error(f"Unexpected error loading data: {e}")
        raise DataLoadingError("An unexpected error occurred during data loading.")

def preprocess_data(data):
    try:
        y = data['target']
        X = data.drop(columns=['target'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except KeyError as e:
        logging.error(f"Missing target column: {e}")
        raise DataLoadingError("Target column is missing from data.")
    except Exception as e:
        logging.error(f"Unexpected error during data preprocessing: {e}")
        raise DataLoadingError("An unexpected error occurred during data preprocessing.")

def train_model(X_train, y_train, model_path='model.xml'):
    try:
        ie = IECore()
        net = ie.read_network(model=model_path)
        exec_net = ie.load_network(network=net, device_name="CPU")

        input_blob = next(iter(net.input_info))
        output_blob = next(iter(net.outputs))

        # Example training loop (pseudo-code)
        for i in range(len(X_train)):
            input_data = X_train.iloc[i].to_numpy()
            output_data = np.array([y_train.iloc[i]])
            exec_net.infer(inputs={input_blob: input_data})
            # Update the model based on output_data (skipping details)
        
        return exec_net
    except FileNotFoundError as e:
        logging.error(f"Model file not found: {e}")
        raise ModelTrainingError("Model file is missing.")
    except Exception as e:
        logging.error(f"Unexpected error during model training: {e}")
        raise ModelTrainingError("An unexpected error occurred during model training.")

def evaluate_model(exec_net, X_test, y_test, model_path='model.xml'):
    try:
        ie = IECore()
        net = ie.read_network(model=model_path)
        input_blob = next(iter(net.input_info))
        output_blob = next(iter(net.outputs))

        predictions = []
        for i in range(len(X_test)):
            input_data = X_test.iloc[i].to_numpy()
            res = exec_net.infer(inputs={input_blob: input_data})
            predictions.append(res[output_blob])

        y_pred = np.array(predictions).flatten()
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logging.info(f"Model accuracy: {accuracy}")
        logging.info(f"Classification report:\n{report}")
    except Exception as e:
        logging.error(f"Unexpected error during model evaluation: {e}")
        raise EvaluationError("An unexpected error occurred during model evaluation.")

def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = "".join(page.get_text() for page in doc)
        logging.info(f"Text extracted from {file_path}")
        return text
    except FileNotFoundError as e:
        logging.error(f"PDF file not found: {e}")
        return None
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs)
        logging.info(f"Text extracted from {file_path}")
        return text
    except FileNotFoundError as e:
        logging.error(f"DOCX file not found: {e}")
        return None
    except Exception as e:
        logging.error(f"Error extracting text from DOCX: {e}")
        return None

def extract_text_from_html(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            text = soup.get_text()
        logging.info(f"Text extracted from {file_path}")
        return text
    except FileNotFoundError as e:
        logging.error(f"HTML file not found: {e}")
        return None
    except Exception as e:
        logging.error(f"Error extracting text from HTML: {e}")
        return None

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file):
            text = file.read()
        logging.info(f"Text extracted from {file_path}")
        return text
    except FileNotFoundError as e:
        logging.error(f"TXT file not found: {e}")
        return None
    except Exception as e:
        logging.error(f"Error extracting text from TXT: {e}")
        return None

def extract_text_from_eml(file_path):
    try:
        import email
        from email import policy
        from email.parser import BytesParser

        with open(file_path, 'rb') as file:
            msg = BytesParser(policy=policy.default).parse(file)

        text = msg.get_body(preferencelist=('plain', 'html')).get_content()
        attachments = []
        for part in msg.iter_attachments():
            file_data = part.get_payload(decode=True)
            attachments.append((part.get_filename(), file_data))
        
        logging.info(f"Text and attachments extracted from {file_path}")
        return text, attachments
    except FileNotFoundError as e:
        logging.error(f"EML file not found: {e}")
        return None, []
    except Exception as e:
        logging.error(f"Error extracting text from EML: {e}")
        return None, []

def process_files_in_parallel(file_paths, max_workers=5):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(extract_text_from_file, file_path): file_path for file_path in file_paths}
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                logging.info(f"File processed successfully: {file_path}")
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
    return results

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext in ['.html', '.htm']:
        return extract_text_from_html(file_path)
    elif ext == '.txt':
        return extract_text_from_txt(file_path)
    elif ext == '.eml':
        return extract_text_from_eml(file_path)
    else:
        logging.warning(f"Unsupported file format: {file_path}")
        return None

def main():
    try:
        config = load_config()
        if not config:
            return

        data = load_data(config['file_path'], config['file_type'])
        if data is None:
            return

        X_train, X_test, y_train, y_test = preprocess_data(data)
        if X_train is None:
            return
        
        model_path = os.path.join(config['project_directory'], 'model.xml')
        trained_model = train_model(X_train, y_train, model_path)
        if trained_model is None:
            return
        
        evaluate_model(trained_model, X_test, y_test, model_path)

        # Example for text extraction
        pdf_text = extract_text_from_pdf("example.pdf")
        docx_text = extract_text_from_docx("example.docx")
        html_text = extract_text_from_html("example.html")
        txt_text = extract_text_from_txt("example.txt")
        eml_text, attachments = extract_text_from_eml("example.eml")

        logging.info(f"Extracted PDF text: {pdf_text}")
        logging.info(f"Extracted DOCX text: {docx_text}")
        logging.info(f"Extracted HTML text: {html_text}")
        logging.info(f"Extracted TXT text: {txt_text}")
        logging.info(f"Extracted EML text: {eml_text}")
        logging.info(f"Extracted EML attachments: {attachments}")

    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")

if __name__ == "__main__":
    main()
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
