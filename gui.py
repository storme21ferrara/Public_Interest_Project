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
from paddleclas import PaddleClas
from blobconverter import from_path
from optimum.intel import IncQuantizationConfig, quantize_dynamic
from torch_ort import ORTModule
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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

def load_config(config_path='model_config.json'):
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        logging.info("Configuration loaded successfully.")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        raise DataLoadingError("Configuration file is missing or invalid.")

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

def named_entity_recognition(text):
    # Placeholder function for named entity recognition
    return [('Australia', 'GPE'), ('Constitution', 'LAW')]

def extract_contextual_keywords(text):
    """Extract context-specific keywords from text."""
    try:
        keywords = re.findall(r'\b(contract|agreement|law|court|judge|legal|constitution|statute|regulation)\b', text, re.I)
        return ', '.join(set(keywords))
    except Exception as e:
        logging.error(f"Error extracting contextual keywords: {e}")
        return ''

def extract_sentiment_over_time(text):
    """Extract sentiment scores over time from text with temporal data."""
    try:
        # Example of extracting sentiment over time (assuming temporal data is present)
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', text)
        sentiments = [TextBlob(text).sentiment.polarity for _ in dates]
        return list(zip(dates, sentiments))
    except Exception as e:
        logging.error(f"Error extracting sentiment over time: {e}")
        return []

def extract_tone_over_time(text):
    """Extract tone scores over time from text with temporal data."""
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
    """Identify relevant statutes and court rules from the text data."""
    try:
        # Example keywords and phrases for identifying statutes
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
    """Extract legislation references specific to Australia."""
    try:
        # Example regex patterns for Australian legislation
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
    """Extract references to the Constitution of the Commonwealth of Australia."""
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
    """Extract additional metadata for legislation."""
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
    """Build a comprehensive DataFrame from the processed file results."""
    try:
        df = pd.DataFrame(results)
        
        # Expand entities
        entity_df = df['entities'].apply(pd.Series).stack().reset_index(level=1, drop=True).apply(pd.Series)
        entity_df.columns = ['entity', 'entity_label']
        df = df.drop(columns=['entities']).join(entity_df, how='left')
        
        # Expand emotions
        emotion_df = df['emotion'].apply(pd.Series).stack().reset_index(level=1, drop=True).apply(pd.Series)
        emotion_df.columns = ['emotion_label', 'emotion_score']
        df = df.drop(columns=['emotion']).join(emotion_df, how='left')
        
        # Expand tones
        tone_df = df['tone'].apply(pd.Series).stack().reset_index(level=1, drop=True).apply(pd.Series)
        tone_df.columns = ['tone_label', 'tone_score']
        df = df.drop(columns(['tone']).join(tone_df, how='left'))
        
        # Extract additional features
        df['Word Count'] = df['preprocessed_text'].apply(lambda x: len(x.split()))
        df['Character Count'] = df['preprocessed_text'].apply(len)
        df['Average Word Length'] = df['preprocessed_text'].apply(lambda x: len(x) / len(x.split()) if len(x.split()) > 0 else 0)
        
        # Add metadata and legal context
        df['Document Type'] = df['file_path'].apply(lambda x: os.path.splitext(x)[-1].lower())
        df['Date'] = pd.to_datetime(df['file_path'].apply(lambda x: re.search(r'\d{4}-\d{2}-\d{2}', x).group() if re.search(r'\d{4}-\d{2}-\d{2}', x) else ''))
        df['Legal Entities'] = df['text'].apply(lambda x: ', '.join(set([ent for ent, label in named_entity_recognition(x) if label in ['ORG', 'LAW', 'GPE']])))
        
        # Add contextual features
        df['Legal References'] = df['text'].apply(extract_legislation_references)
        df['Contextual Keywords'] = df['text'].apply(extract_contextual_keywords)
        
        # Add sentiment and tone over time
        df['Sentiment Over Time'] = df['text'].apply(extract_sentiment_over_time)
        df['Tone Over Time'] = df['text'].apply(extract_tone_over_time)
        
        # Add topic modeling and similarity analysis
        df['Topics'] = extract_topics(df['text'].tolist())
        df['Similarity Matrix'] = [calculate_similarity(df['text'].tolist())]
        
        # Add sentiment and tone distribution
        df['Sentiment Distribution'] = df['text'].apply(sentiment_distribution)
        df['Tone Distribution'] = df['text'].apply(tone_distribution)
        
        # Add entity relations and graph analysis
        df['Entity Relations'] = df['text'].apply(extract_entity_relations)
        df['Entity Graph'] = df['Entity Relations'].apply(build_entity_graph)
        
        # Add relevant statutes identification
        df['Relevant Statutes'] = df['text'].apply(identify_relevant_statutes)
        
        logging.info("Main DataFrame built successfully.")
        
        return df
    except Exception as e:
        logging.error(f"Error building main DataFrame: {e}")
        return None

def build_legislation_dataframe(df):
    """Build a legislation DataFrame from the text DataFrame."""
    try:
        df['Legislation References'] = df['text'].apply(extract_legislation_references)
        df['Constitution References'] = df['text'].apply(extract_constitution_references)
        df['Legislation Metadata'] = df['text'].apply(extract_legislation_metadata)
        
        leg_df = df[['file_path', 'Legislation References', 'Constitution References', 'Legislation Metadata']]
        leg_df['Commonwealth'] = leg_df['Legislation References'].apply(lambda x: x['Commonwealth'])
        leg_df['States'] = leg_df['Legislation References'].apply(lambda x: ', '.join(x['State']))
        leg_df['Constitution'] = leg_df['Constitution References'].apply(lambda x: x['Constitution'])
        leg_df['Constitution Sections'] = leg_df['Constitution References'].apply(lambda x: ', '.join(x['Sections']))
        leg_df['Jurisdiction'] = leg_df['Legislation Metadata'].apply(lambda x: x['Jurisdiction'])
        leg_df['Title'] = leg_df['Legislation Metadata'].apply(lambda x: x['Title'])
        leg_df['Amendments'] = leg_df['Legislation Metadata'].apply(lambda x: x['Amendments'])
        leg_df = leg_df.drop(columns(['Legislation References', 'Constitution References', 'Legislation Metadata']))
        
        logging.info("Legislation DataFrame built successfully.")
        return leg_df
    except Exception as e:
        logging.error(f"Error building legislation DataFrame: {e}")
        return None

def build_all_dataframes(results):
    """Build main and legislation DataFrames."""
    main_df = build_main_dataframe(results)
    legislation_df = build_legislation_dataframe(main_df)
    
    return main_df, legislation_df

def visualize_sentiment_distribution(sentiments):
    """Visualize sentiment distribution."""
    try:
        sns.histplot(sentiments, kde=True)
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Frequency')
        plt.show()
    except Exception as e:
        logging.error(f"Error visualizing sentiment distribution: {e}")

def visualize_tone_distribution(tones):
    """Visualize tone distribution."""
    try:
        tone_scores = [tone['score'] for tone in tones if 'score' in tone]
        sns.histplot(tone_scores, kde=True)
        plt.title('Tone Distribution')
        plt.xlabel('Tone Score')
        plt.ylabel('Frequency')
        plt.show()
    except Exception as e:
        logging.error(f"Error visualizing tone distribution: {e}")

class ProjectGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Project GUI")

        self.file_paths = []
        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self.root, text="Select files for processing:")
        self.label.pack(pady=10)

        self.select_button = tk.Button(self.root, text="Select Files", command=self.select_files)
        self.select_button.pack(pady=5)

        self.process_button = tk.Button(self.root, text="Process Files", command=self.process_files)
        self.process_button.pack(pady=20)

    def select_files(self):
        self.file_paths = filedialog.askopenfilenames()
        if self.file_paths:
            messagebox.showinfo("Selected Files", "\n".join(self.file_paths))
        else:
            messagebox.showwarning("No Files Selected", "Please select at least one file.")

    def process_files(self):
        if not self.file_paths:
            messagebox.showwarning("No Files Selected", "Please select at least one file.")
            return

        try:
            results = process_files_in_parallel(self.file_paths)
            main_df, legislation_df = build_all_dataframes(results)

            if main_df is not None:
                main_df.to_csv('main_dataframe.csv', index=False)
                messagebox.showinfo("Success", "Main DataFrame saved to main_dataframe.csv")

            if legislation_df is not None:
                legislation_df.to_csv('legislation_dataframe.csv', index=False)
                messagebox.showinfo("Success", "Legislation DataFrame saved to legislation_dataframe.csv")
        except Exception as e:
            logging.error(f"Error processing files: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ProjectGUI(root)
    root.mainloop()
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
