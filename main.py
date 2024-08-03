import os
import re
import logging
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import docx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from openvino.inference_engine import IECore
from transformers import pipeline
from textblob import TextBlob

# Import PaddleOCR with error handling for protobuf issue
try:
    from paddleocr import PaddleOCR
except ImportError as e:
    logging.error(f"Failed to import PaddleOCR: {e}. Attempting to resolve the issue...")
    os.system("pip install protobuf==3.20.3")
    from paddleocr import PaddleOCR

from paddleclas import PaddleClas

# Try to import from_path from blobconverter
try:
    from blobconverter import from_path
except ImportError as e:
    logging.error(f"Failed to import 'from_path' from 'blobconverter': {e}. Please ensure 'blobconverter' is installed and up to date.")
    from_path = None  # Define a fallback if possible

# Try to import IncQuantizationConfig and quantize_dynamic from optimum.intel
try:
    from optimum.intel import IncQuantizationConfig, quantize_dynamic
except ImportError as e:
    logging.error(f"Failed to import 'IncQuantizationConfig' and 'quantize_dynamic' from 'optimum.intel': {e}. Please ensure 'optimum[intel]' is installed and up to date.")
    IncQuantizationConfig = None
    quantize_dynamic = None

try:
    from torch_ort import ORTModule
except ImportError as e:
    logging.error(f"Failed to import 'ORTModule' from 'torch_ort': {e}. Please ensure 'torch_ort' is installed and up to date.")
    ORTModule = None

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Import MultiSpeakerTTSModel from deepvoice3_pytorch
from deepvoice3_pytorch import MultiSpeakerTTSModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class CustomError(Exception):
    """Base class for custom exceptions"""
    pass

class DataLoadingError(CustomError):
    """Exception raised for errors in the data loading process."""
    pass

class ModelTrainingError(CustomError):
    """Exception raised for errors during model training."""
    pass

class EvaluationError(CustomError):
    """Exception raised for errors during model evaluation."""
    pass

def load_config(config_path='model_config.json'):
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

def load_pretrained_models(config_path='model_config.json'):
    try:
        config = load_config(config_path)
        ie = IECore()
        models = {}
        for model_name, model_path in config['openvino_models'].items():
            net = ie.read_network(model=model_path)
            exec_net = ie.load_network(network=net, device_name="CPU")
            models[model_name] = exec_net
        logging.info("OpenVINO models loaded successfully.")

        paddle_ocr = PaddleOCR()
        logging.info("PaddleOCR initialized successfully.")

        voice_model = MultiSpeakerTTSModel(model_path=config['deepvoice_model_path'])
        logging.info("DeepVoice model initialized successfully.")
        
        nlp_model = pipeline("ner", model=config['transformer_model'])
        logging.info("Hugging Face Transformers model initialized successfully.")

        return models, paddle_ocr, voice_model, nlp_model
        
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        raise DataLoadingError("Configuration file is missing.")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from configuration file: {e}")
        raise DataLoadingError("Error decoding JSON configuration.")
    except Exception as e:
        logging.error(f"Error during model initialization: {e}")
        raise DataLoadingError(f"An unexpected error occurred during model initialization: {e}")

def extract_text_with_ocr(file_path, ocr_engine):
    try:
        if ocr_engine == 'paddle':
            ocr = PaddleOCR()
        elif ocr_engine == 'rapid':
            ocr = OCR()
        else:
            raise ValueError("Unsupported OCR engine specified.")
        
        result = ocr.ocr(file_path)
        text = ' '.join([line[-1] for line in result])
        logging.info(f"Text extracted from {file_path} using {ocr_engine} OCR.")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {e}")
        return None

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

def advanced_model_training(X_train, y_train):
    try:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=StratifiedKFold(n_splits=5), n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        logging.info(f"Best Model Parameters: {grid_search.best_params_}")
        return best_model
    except Exception as e:
        logging.error(f"Error during advanced model training: {e}")
        raise ModelTrainingError(f"An unexpected error occurred during model training: {e}")

def visualize_data(data):
    try:
        sns.pairplot(data)
        plt.show()
        logging.info("Data visualization completed.")
    except Exception as e:
        logging.error(f"Error visualizing data: {e}")

def visualize_relationships(data):
    try:
        graph = visualize_relationships(data)
        graph.render('output/relationships', view=True)
        logging.info("Relationship visualization completed.")
    except Exception as e:
        logging.error(f"Error visualizing relationships: {e}")

def clean_data(data):
    data.dropna(inplace=True)
    return data

def handle_missing_values(data):
    data.fillna(data.mean(), inplace=True)
    return data

def handle_outliers(data):
    for col in data.select_dtypes(include=["float64", "int64"]).columns:
        q1 = data[col]
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data

def normalize_data(data):
    for col in data.select_dtypes(include=["float64", "int64"]).columns:
        data[col] = (data[col] - data[col].mean()) / data[col].std()
    return data

def extract_features(data):
    # Placeholder for actual feature extraction logic
    return data

def process_files_in_parallel(file_paths, max_workers=5):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(extract_text_with_ocr, file_path, 'paddle'): file_path for file_path in file_paths}
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                logging.info(f"File processed successfully: {file_path}")
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
    return results

def named_entity_recognition(text):
    # Placeholder function for named entity recognition
    return [('Australia', 'GPE'), ('Constitution', 'LAW')]

def extract_contextual_keywords(text):
    try:
        keywords = re.findall(r'\b(contract|agreement|law|court|judge|legal|constitution|statute|regulation)\b', text, re.I)
        return ', '.join(set(keywords))
    except Exception as e:
        logging.error(f"Error extracting contextual keywords: {e}")
        return ''

def extract_sentiment_over_time(text):
    try:
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', text)
        sentiments = [TextBlob(text).sentiment.polarity for _ in dates]
        return list(zip(dates, sentiments))
    except Exception as e:
        logging.error(f"Error extracting sentiment over time: {e}")
        return []

def extract_tone_over_time(text):
    try:
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', text)
        tones = [TextBlob(text).sentiment.subjectivity for _ in dates]
        return list(zip(dates, tones))
    except Exception as e:
        logging.error(f"Error extracting tone over time: {e}")
        return []

def extract_topics(texts):
    # Placeholder function for extracting topics
    return []

def calculate_similarity(texts):
    # Placeholder function for calculating similarity matrix
    return []

def sentiment_distribution(text):
    # Placeholder function for sentiment distribution
    return []

def tone_distribution(text):
    # Placeholder function for tone distribution
    return []

def extract_entity_relations(text):
    # Placeholder function for extracting entity relations
    return []

def build_entity_graph(relations):
    # Placeholder function for building entity graph
    return []

def identify_relevant_statutes(text):
    try:
        keywords = [
            'Constitution of the Commonwealth of Australia',
            'Commonwealth Act', 'State Act', 'Regulation', 'Rule', 'Statute', 'Section'
        ]
        relevant_statutes = [kw for kw in keywords if kw.lower() in text.lower()]
        return ', '.join(relevant_statutes)
    except Exception as e:
        logging.error(f"Error identifying relevant statutes: {e}")
        return ''

def extract_legislation_references(text):
    try:
        commonwealth_pattern = re.compile(r'\b(Cth|Commonwealth|Federal)\b', re.IGNORECASE)
        state_patterns = {
            'NSW': re.compile(r'\b(NSW|New South Wales)\b', re.IGNORECASE),
            'VIC': re.compile(r'\b(VIC|Victoria)\b', re.IGNORECASE),
            'QLD': re.compile(r'\b(QLD|Queensland)\b', re.IGNORECASE),
            'WA': re.compile(r'\b(WA|Western Australia)\b', re.IGNORECASE),
            'SA': re.compile(r'\b(SA|South Australia)\b', re.IGNORECASE),
            'TAS': re.compile(r'\b(TAS|Tasmania)\b', re.IGNORECASE),
            'ACT': re.compile(r'\b(ACT|Australian Capital Territory)\b', re.IGNORECASE),
            'NT': re.compile(r'\b(NT|Northern Territory)\b', re.IGNORECASE)
        }
        references = {
            'Commonwealth': bool(commonwealth_pattern.search(text)),
            'State': []
        }
        for state, pattern in state_patterns.items():
            if pattern.search(text):
                references['State'].append(state)
        return references
    except Exception as e:
        logging.error(f"Error extracting legislation references: {e}")
        return {'Commonwealth': False, 'State': []}

def extract_constitution_references(text):
    try:
        constitution_pattern = re.compile(r'\bConstitution\b', re.IGNORECASE)
        sections = re.findall(r'Section\s\d+', text, re.IGNORECASE)
        references = {
            'Constitution': bool(constitution_pattern.search(text)),
            'Sections': sections
        }
        return references
    except Exception as e:
        logging.error(f"Error extracting constitution references: {e}")
        return {'Constitution': False, 'Sections': []}

def extract_legislation_metadata(text):
    try:
        jurisdiction = re.search(r'\b(NSW|VIC|QLD|WA|SA|TAS|ACT|NT|Commonwealth)\b', text, re.IGNORECASE)
        title = re.search(r'\b(?:Act|Regulation|Rule|Statute)\b.*?\b\d{4}\b', text, re.IGNORECASE)
        amendments = re.findall(r'\b(?:Amendment|Repeal|Insert)\b', text, re.IGNORECASE)
        
        metadata = {
            'Jurisdiction': jurisdiction.group() if jurisdiction else '',
            'Title': title.group() if title else '',
            'Amendments': ', '.join(amendments)
        }
        return metadata
    except Exception as e:
        logging.error(f"Error extracting legislation metadata: {e}")
        return {'Jurisdiction': '', 'Title': '', 'Amendments': ''}

def build_main_dataframe(results):
    try:
        df = pd.DataFrame(results)
        
        entity_df = df['entities'].apply(pd.Series).stack().reset_index(level=1, drop=True).apply(pd.Series)
        entity_df.columns = ['entity', 'entity_label']
        df = df.drop(columns=['entities']).join(entity_df, how='left')
        
        emotion_df = df['emotion'].apply(pd.Series).stack().reset_index(level=1, drop=True).apply(pd.Series)
        emotion_df.columns = ['emotion_label', 'emotion_score']
        df = df.drop(columns=['emotion']).join(emotion_df, how='left')
        
        tone_df = df['tone'].apply(pd.Series).stack().reset_index(level=1, drop=True).apply(pd.Series)
        tone_df.columns = ['tone_label', 'tone_score']
        df = df.drop(columns(['tone']).join(tone_df, how='left'))
        
        df['Word Count'] = df['preprocessed_text'].apply(lambda x: len(x.split()))
        df['Character Count'] = df['preprocessed_text'].apply(len)
        df['Average Word Length'] = df['preprocessed_text'].apply(lambda x: len(x) / len(x.split()) if len(x.split()) > 0 else 0)
        
        df['Document Type'] = df['file_path'].apply(lambda x: os.path.splitext(x)[-1].lower())
        df['Date'] = pd.to_datetime(df['file_path'].apply(lambda x: re.search(r'\d{4}-\d{2}-\d{2}', x).group() if re.search(r'\d{4}-\d{2}-\d{2}', x) else ''))
        df['Legal Entities'] = df['text'].apply(lambda x: ', '.join(set([ent for ent, label in named_entity_recognition(x) if label in ['ORG', 'LAW', 'GPE']])))
        
        df['Legal References'] = df['text'].apply(extract_legislation_references)
        df['Contextual Keywords'] = df['text'].apply(extract_contextual_keywords)
        
        df['Sentiment Over Time'] = df['text'].apply(extract_sentiment_over_time)
        df['Tone Over Time'] = df['text'].apply(extract_tone_over_time)
        
        df['Topics'] = extract_topics(df['text'].tolist())
        df['Similarity Matrix'] = [calculate_similarity(df['text'].tolist())]
        
        df['Sentiment Distribution'] = df['text'].apply(sentiment_distribution)
        df['Tone Distribution'] = df['text'].apply(tone_distribution)
        
        df['Entity Relations'] = df['text'].apply(extract_entity_relations)
        df['Entity Graph'] = df['Entity Relations'].apply(build_entity_graph)
        
        df['Relevant Statutes'] = df['text'].apply(identify_relevant_statutes)
        
        logging.info("Main DataFrame built successfully.")
        
        return df
    except Exception as e:
        logging.error(f"Error building main DataFrame: {e}")
        return None

def build_legislation_dataframe(df):
    try:
        df['Legislation References'] = df['text'].apply(extract_legislation_references)
        df['Constitution References'] = df['text'].apply(extract_constitution_references)
        df['Legislation Metadata'] = df['text'].apply(extract_legislation_metadata)

        legislation_df = pd.DataFrame({
            'File Path': df['file_path'],
            'Legislation References': df['Legislation References'],
            'Constitution References': df['Constitution References'],
            'Legislation Metadata': df['Legislation Metadata']
        })

        logging.info("Legislation DataFrame built successfully.")
        return legislation_df
    except Exception as e:
        logging.error(f"Error building legislation DataFrame: {e}")
        return None

def main():
    try:
        config = load_config()
        
        data_path = config['data_path']
        file_paths = config['file_paths']
        model_save_path = config['model_save_path']
        logging_level = config['logging_level']
        pretrained_models = config['pretrained_models']

        logging.basicConfig(level=logging_level)
        
        data = load_data(data_path)
        X_train, X_test, y_train, y_test = preprocess_data(data)
        
        model = advanced_model_training(X_train, y_train)
        save_model(model, model_save_path)
        
        results = process_files_in_parallel(file_paths)
        
        main_df = build_main_dataframe(results)
        legislation_df = build_legislation_dataframe(main_df)
        
        visualize_data(main_df)
        visualize_relationships(main_df)
        
    except CustomError as e:
        logging.error(f"Custom error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
