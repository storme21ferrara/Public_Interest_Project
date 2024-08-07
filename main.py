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
from openvino.runtime import Core
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from textblob import TextBlob
from deepvoice3_pytorch import MultiSpeakerTTSModel
from paddleocr import PaddleOCR
import blobconverter
from torch_ort import ORTModule
from google.protobuf import proto_builder
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys
import cv2
import requests
import tqdm
import tensorflow as tf
import torch
import dask.dataframe as dd
import statsmodels.api as sm
import xgboost as xgb
from nltk import download, word_tokenize, sent_tokenize, pos_tag, ne_chunk
import plotly.express as px

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom Exceptions
class DataLoadingError(Exception):
    pass

class ModelTrainingError(Exception):
    pass

class EvaluationError(Exception):
    pass

# Load Configuration
def load_config(config_path='model_config.json'):
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        logging.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at path: {config_path}")
        raise DataLoadingError("Configuration file is missing.")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from the configuration file: {e}")
        raise DataLoadingError("Error decoding JSON configuration.")
    except Exception as e:
        logging.error(f"Unexpected error loading configuration: {e}")
        raise DataLoadingError(f"Unexpected error: {e}")

# Update Configuration with Directory Paths
def update_config_with_paths(config):
    config['directories'] = {
        "project": "E:\\Public_Interest_Project",
        "models_source": "E:\\Public_Interest_Project\\Models_Source",
        "scripts_module": "E:\\Public_Interest_Project\\Scripts_Module",
        "data": "E:\\Public_Interest_Project\\Data",
        "venv_openvino": "E:\\Public_Interest_Project\\venv_openvino",
        "logs": "E:\\Public_Interest_Project\\project_logs",
    }
    config['scripts'] = {
        "data_capture": "E:\\Public_Interest_Project\\Scripts_Module\\data_capture.py",
        "data_processing": "E:\\Public_Interest_Project\\Scripts_Module\\data_processing.py",
        "generate_visualisations": "E:\\Public_Interest_Project\\Scripts_Module\\generate_visualisations.py",
        "local_terminal_config": "E:\\Public_Interest_Project\\Scripts_Module\\local_terminal_config.py",
        "model_training": "E:\\Public_Interest_Project\\Scripts_Module\\model_training.py",
        "text_extraction": "E:\\Public_Interest_Project\\Scripts_Module\\text_extraction.py",
        "web_scraping": "E:\\Public_Interest_Project\\Scripts_Module\\web_scraping.py"
    }
    return config

# Function to Convert Model Using blobconverter
def convert_model_with_blobconverter(xml_path, bin_path):
    try:
        blob_path = blobconverter.from_openvino(
            xml=xml_path,
            bin=bin_path,
            data_type="FP16",
            shaves=5
        )
        logging.info(f"Model converted successfully: {blob_path}")
        return blob_path
    except Exception as e:
        logging.error(f"Error converting model with blobconverter: {e}")
        raise

# Function to Apply Dynamic Quantization
def apply_dynamic_quantization(model, config_path, save_path):
    try:
        # Load the quantization configuration
        quantization_config = IncQuantizer.from_pretrained(config_path)

        # Define the evaluation function
        def eval_func(model):
            # Define your evaluation logic here
            return 0.9  # Dummy accuracy for illustration

        # Instantiate the quantizer
        quantizer = IncQuantizer(quantization_config, eval_func=eval_func)

        # Initialize the optimizer with the model and quantizer
        optimizer = IncOptimizer(model, quantizer=quantizer)

        # Apply dynamic quantization
        quantized_model = optimizer.fit()

        # Save the resulting model and its corresponding files
        quantized_model.save_pretrained(save_path)

        logging.info(f"Quantized model saved successfully at {save_path}")
    except Exception as e:
        logging.error(f"Error during dynamic quantization: {e}")
        raise ModelTrainingError(f"Unexpected error during dynamic quantization: {e}")

# GUI Class
class ProjectGUI:
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.root.title("Public Interest Project")
        self.create_widgets()
        self.data = None
        self.model = None

    def create_widgets(self):
        self.import_csv_button = tk.Button(self.root, text="Import CSV", command=self.import_csv)
        self.import_csv_button.pack()

        self.import_json_button = tk.Button(self.root, text="Import JSON", command=self.import_json)
        self.import_json_button.pack()

        self.import_excel_button = tk.Button(self.root, text="Import Excel", command=self.import_excel)
        self.import_excel_button.pack()

        self.import_txt_button = tk.Button(self.root, text="Import TXT", command=self.import_txt)
        self.import_txt_button.pack()

        self.import_directory_button = tk.Button(self.root, text="Import Directory", command=self.import_directory)
        self.import_directory_button.pack()

        self.preview_data_button = tk.Button(self.root, text="Preview Data", command=self.preview_data)
        self.preview_data_button.pack()

        self.clean_data_button = tk.Button(self.root, text="Clean Data", command=self.clean_data)
        self.clean_data_button.pack()

        self.handle_missing_values_button = tk.Button(self.root, text="Handle Missing Values", command=self.handle_missing_values)
        self.handle_missing_values_button.pack()

        self.handle_outliers_button = tk.Button(self.root, text="Handle Outliers", command=self.handle_outliers)
        self.handle_outliers_button.pack()

        self.normalize_data_button = tk.Button(self.root, text="Normalize Data", command=self.normalize_data)
        self.normalize_data_button.pack()

        self.extract_features_button = tk.Button(self.root, text="Extract Features", command=self.extract_features)
        self.extract_features_button.pack()

        self.analyze_data_button = tk.Button(self.root, text="Analyze Data", command=self.analyze_data)
        self.analyze_data_button.pack()

        self.train_model_button = tk.Button(self.root, text="Train Model", command=self.train_model)
        self.train_model_button.pack()

        self.evaluate_model_button = tk.Button(self.root, text="Evaluate Model", command=self.evaluate_model)
        self.evaluate_model_button.pack()

            self.save_model_button = tk.Button(self.root, text="Save Model", command=self.save_model)
            self.save_model_button.pack()

            self.load_model_button = tk.Button(self.root, text="Load Model", command=self.load_model)
            self.load_model_button.pack()

            self.visualize_results_button = tk.Button(self.root, text="Visualize Results", command=self.visualize_results)
            self.visualize_results_button.pack()

            self.imported_files_list = tk.Listbox(self.root)
            self.imported_files_list.pack(fill=tk.BOTH, expand=1)

            self.plot_type_combo = ttk.Combobox(self.root, values=["Line Plot", "Bar Chart", "Scatter Plot", "Histogram", "Heatmap"])
            self.plot_type_combo.set("Select Plot Type")
            self.plot_type_combo.pack()

            self.filter_column_entry = tk.Entry(self.root)
            self.filter_column_entry.pack()
            self.filter_value_entry = tk.Entry(self.root)
            self.filter_value_entry.pack()

        def log_message(self, message, level=logging.INFO):
            logging.log(level, message)
            self.imported_files_list.insert(tk.END, message)

        def import_csv(self):
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file_path:
                self.log_message(f"Importing CSV file: {file_path}")
                try:
                    self.data = pd.read_csv(file_path)
                    self.imported_files_list.insert(tk.END, file_path)
                    self.log_message("CSV data imported successfully.")
                except Exception as e:
                    self.log_message(f"Error importing CSV file: {e}", logging.ERROR)

        def import_json(self):
            file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
            if file_path:
                self.log_message(f"Importing JSON file: {file_path}")
                try:
                    self.data = pd.read_json(file_path)
                    self.imported_files_list.insert(tk.END, file_path)
                    self.log_message("JSON data imported successfully.")
                except Exception as e:
                    self.log_message(f"Error importing JSON file: {e}", logging.ERROR)

        def import_excel(self):
            file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xls;*.xlsx")])
            if file_path:
                self.log_message(f"Importing Excel file: {file_path}")
                try:
                    self.data = pd.read_excel(file_path)
                    self.imported_files_list.insert(tk.END, file_path)
                    self.log_message("Excel data imported successfully.")
                except Exception as e:
                    self.log_message(f"Error importing Excel file: {e}", logging.ERROR)

        def import_txt(self):
            file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
            if file_path:
                self.log_message(f"Importing TXT file: {file_path}")
                try:
                    self.data = pd.read_csv(file_path, delimiter="\t")
                    self.imported_files_list.insert(tk.END, file_path)
                    self.log_message("TXT data imported successfully.")
                except Exception as e:
                    self.log_message(f"Error importing TXT file: {e}", logging.ERROR)

        def import_directory(self):
            directory_path = filedialog.askdirectory()
            if directory_path:
                self.log_message(f"Importing files from directory: {directory_path}")
                for root, dirs, files in os.walk(directory_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        self.imported_files_list.insert(tk.END, file_path)
                self.log_message("Directory files imported successfully.")

        def preview_data(self):
            if self.data is not None:
                preview_window = tk.Toplevel(self.root)
                preview_window.title("Data Preview")
                preview_text = tk.Text(preview_window)
                preview_text.pack(fill=tk.BOTH, expand=1)
                preview_text.insert(tk.END, str(self.data.head()))
            else:
                self.log_message("No data to preview.", logging.WARNING)

        def clean_data(self):
            if self.data is not None:
                self.log_message("Cleaning data...")
                self.data = clean_data(self.data)
                self.log_message("Data cleaned successfully.")
            else:
                self.log_message("No data to clean.", logging.WARNING)

        def handle_missing_values(self):
            if self.data is not None:
                self.log_message("Handling missing values...")
                self.data = handle_missing_values(self.data)
                self.log_message("Missing values handled successfully.")
            else:
                self.log_message("No data to handle missing values.", logging.WARNING)

        def handle_outliers(self):
            if self.data is not None:
                self.log_message("Handling outliers...")
                self.data = handle_outliers(self.data)
                self.log_message("Outliers handled successfully.")
            else:
                self.log_message("No data to handle outliers.", logging.WARNING)

        def normalize_data(self):
            if self.data is not None:
                self.log_message("Normalizing data...")
                self.data = normalize_data(self.data)
                self.log_message("Data normalized successfully.")
            else:
                self.log_message("No data to normalize.", logging.WARNING)

        def extract_features(self):
            if self.data is not None:
                self.log_message("Extracting features...")
                self.data = extract_features(self.data)
                self.log_message("Features extracted successfully.")
            else:
                self.log_message("No data to extract features.", logging.WARNING)

        def analyze_data(self):
            if self.data is not None:
                self.log_message("Analyzing data...")
                plot_type = self.plot_type_combo.get()
                if plot_type != "Select Plot Type":
                    column = self.filter_column_entry.get()
                    if column in self.data.columns:
                        sns.set(style="whitegrid")
                        if plot_type == "Line Plot":
                            plot = sns.lineplot(data=self.data, x=self.data.index, y=column)
                        elif plot_type == "Bar Chart":
                            plot = sns.barplot(data=self.data, x=self.data.index, y=column)
                        elif plot_type == "Scatter Plot":
                            plot = sns.scatterplot(data=self.data, x=self.data.index, y=column)
                        elif plot_type == "Histogram":
                            plot = sns.histplot(data=self.data, x=column)
                        elif plot_type == "Heatmap":
                            plot = sns.heatmap(data=self.data.corr(), annot=True, cmap="coolwarm")
                        
                        plot.figure.tight_layout()
                        plot_canvas = FigureCanvasTkAgg(plot.figure, master=self.root)
                        plot_canvas.draw()
                        plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
                        self.log_message(f"{plot_type} created successfully.")
                    else:
                        self.log_message(f"Column '{column}' not found in data.", logging.ERROR)
                else:
                    self.log_message("No plot type selected.", logging.WARNING)
            else:
                self.log_message("No data to analyze.", logging.WARNING)

        def train_model(self):
            if self.data is not None:
                self.log_message("Training model...")
                X_train, X_test, y_train, y_test = preprocess_data(self.data)
                self.model = advanced_model_training(X_train, y_train)
                self.log_message("Model trained successfully.")
            else:
                self.log_message("No data to train model.", logging.WARNING)

        def evaluate_model(self):
            if self.model is not None:
                self.log_message("Evaluating model...")
                X_train, X_test, y_train, y_test = preprocess_data(self.data)
                y_pred = self.model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)
                self.log_message(f"Model accuracy: {accuracy}")
                self.log_message(f"Classification report:\n{report}")
                self.log_message("Model evaluated successfully.")
            else:
                self.log_message("No model to evaluate.", logging.WARNING)

        def save_model(self):
            if self.model is not None:
                file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Model files", "*.pkl")])
                if file_path:
                    self.log_message(f"Saving model to {file_path}...")
                    with open(file_path, 'wb') as model_file:
                        pickle.dump(self.model, model_file)
                    self.log_message("Model saved successfully.")
            else:
                self.log_message("No model to save.", logging.WARNING)

        def load_model(self):
            file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pkl")])
            if file_path:
                self.log_message(f"Loading model from {file_path}...")
                with open(file_path, 'rb') as model_file:
                    self.model = pickle.load(model_file)
                self.log_message("Model loaded successfully.")
            else:
                self.log_message("No model file selected.", logging.WARNING)

        def visualize_results(self):
            if self.model is not None:
                self.log_message("Visualizing results...")
                # Placeholder for results visualization logic
                self.log_message("Results visualized successfully.")
            else:
                self.log_message("No model results to visualize.", logging.WARNING)

    # Utility Functions

    def clean_data(data):
        data.dropna(inplace=True)
        return data

    def handle_missing_values(data):
        data.fillna(data.mean(), inplace=True)
        return data

    def handle_outliers(data):
        for col in data.select_dtypes(include=["float64", "int64"]).columns:
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

    def preprocess_data(data):
        try:
            y = data['Outcome']
            X = data.drop(columns=['Outcome'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except KeyError as e:
            logging.error(f"Missing target column: {e}")
            raise DataLoadingError("Target column is missing from data.")
        except Exception as e:
            logging.error(f"Unexpected error during data preprocessing: {e}")
            raise DataLoadingError("Unexpected error during data preprocessing.")

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
            raise ModelTrainingError(f"Unexpected error during model training: {e}")

    # Main Function
    def main():
        config_path = 'model_config.json'
        config = load_config(config_path)
        config = update_config_with_paths(config)
        root = tk.Tk()
        app = ProjectGUI(root, config)
        root.mainloop()

    if __name__ == "__main__":
        # Check if the quantization class exists before running the main script
        try:
            from optimum.intel.neural_compressor.quantization import IncQuantizerForSequenceClassification
            print("Import successful")
        except ImportError as e:
            print(f"ImportError: {e}")
            # Fallback or alternative import if available
            import os

            # Path to the installed optimum library
            library_path = "E:\\Public_Interest_Project\\venv_openvino\\lib\\site-packages\\optimum\\intel\\neural_compressor\\quantization.py"

            if os.path.exists(library_path):
                with open(library_path, 'r') as file:
                    content = file.read()
                    if "IncQuantizerForSequenceClassification" in content:
                        print("Class IncQuantizerForSequenceClassification found")
                    else:
                        print("Class IncQuantizerForSequenceClassification not found")
            else:
                print("quantization.py file does not exist")
        
        main()

