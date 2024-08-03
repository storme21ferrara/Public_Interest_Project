import os
import re
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import docx
import matplotlib.pyplot as plt
import seaborn as sns
from rapidocr_pdf import PDFExtracter
from paddleocr import PaddleOCR
from paddleocr_convert import convert_ocr_result
from infbench import benchmark
from artificialvision import analyze_image
from modelconv import convert_model
from optimum_benchmark import benchmark_optimization
from visiongraph import visualize_graph
from geti_sdk import Geti
from tflite2tensorflow import convert
from anomalib import detect_anomalies
from vector_forge import process_vectors
from openvino_genai import generate_model
from torch_ort_infer import infer_with_onnx_runtime
from openvino_model_api import OpenVINOModel
from nncf import compress_model
from blobconverter import convert_to_blob
from optimum_intel import optimize_model_for_intel
from otx import train_model
from ovmsclient import make_prediction
from llama_index_llms_openvino import use_llama_model
from llama_index_embeddings_openvino import generate_embeddings
import cv2
from rapidocr_openvinogpu import GPUOCR
from optimum import optimize
from rapidocr_openvino import OpenVINO_OCR
from mlserver_openvino import OpenVINOModel as MLServerOpenVINOModel
from openvino_kaggle import KaggleModel
from openvino_optimum import optimize_openvino_model
from openvino_tokenizers import tokenize_text
from openvino_workbench import Workbench
from openvino2onnx import convert_model as convert_openvino_to_onnx
from openvino_dev import develop_model
from rapid_layout import analyze_layout
from transformers import pipeline
from textblob import TextBlob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(filename='human_behaviour_analysis.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def load_pipelines():
    """
    Load NLP pipelines for behavior analysis.
    
    :return: Dictionary containing loaded NLP pipelines.
    """
    try:
        return {
            'icd_11_pipeline': pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english"),
            'sentiment_pipeline': pipeline("sentiment-analysis"),
            'summarization_pipeline': pipeline("summarization"),
            'ner_pipeline': pipeline("ner", grouped_entities=True),
            'qa_pipeline': pipeline("question-answering"),
            'classifier_pipeline': pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
        }
    except Exception as e:
        logging.error(f"Error loading NLP models: {e}")
        return {}

pipelines = load_pipelines()

# Regex patterns for ABN and ACN
ABN_PATTERN = re.compile(r'\b\d{2}\s\d{3}\s\d{3}\s\d{3}\b')
ACN_PATTERN = re.compile(r'\b\d{3}\s\d{3}\s\d{3}\b')

# Psychological and malicious behavior phrases
MANIPULATION_PHRASES = [
    "You're being dramatic", "That's not true", "Where did you get a crazy idea like that?",
    "It's all your fault", "I wouldn't have messed up if you hadn't upset me"
]
GASLIGHTING_PHRASES = [
    "I didn't do that", "You never saw those texts", "You're crazy",
    "No one will ever believe you", "Calm down", "Quit overreacting",
    "Look what you made me do"
]
CORRUPT_CONDUCT_PHRASES = [
    "bribery", "fraud", "coercion", "intimidation", "unlawful"
]
UNLAWFUL_CONDUCT_PHRASES = [
    "illegal", "unauthorized", "unlawful", "criminal", "felony", "misdemeanor"
]

def query_icd_api(icd_code):
    """
    Query the ICD API with a given ICD code.

    :param icd_code: ICD code to query.
    :return: API response as a string.
    """
    try:
        result = subprocess.run(["C:\\Program Files\\ICD-API\\ICD-API.exe", icd_code], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        logging.error(f"Error querying ICD API: {e}")
        return ""

def compare_traits(icd_data, dsm_data):
    """
    Compare traits between ICD-11 and DSM-5 data.

    :param icd_data: DataFrame containing ICD-11 data.
    :param dsm_data: DataFrame containing DSM-5 data.
    :return: Dictionary comparing the traits.
    """
    try:
        comparison = {}
        for trait in icd_data.columns:
            comparison[trait] = {
                'ICD-11': icd_data[trait].mean(),
                'DSM-5': dsm_data.get(trait, pd.Series()).mean()  # Handle missing traits
            }
        return comparison
    except Exception as e:
        logging.error(f"Error comparing traits: {e}")
        return {}

def load_and_prepare_data(file_path):
    """
    Load and prepare data from a CSV file.

    :param file_path: Path to the CSV file.
    :return: Prepared DataFrame or None if an error occurs.
    """
    if not os.path.exists(file_path):
        logging.error(f"No such file or directory: '{file_path}'")
        return None
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            logging.error("No columns to parse from file")
            return None
        data.fillna(method='ffill', inplace=True)  # Handle missing values
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        logging.info(f"Data loaded and prepared from {file_path}.")
        return pd.DataFrame(scaled_data, columns=data.columns)
    except Exception as e:
        logging.error(f"Error loading and preparing data: {e}")
        return None

def extract_personality_traits(data, model='DSM-5'):
    """
    Extract personality traits from the data.

    :param data: DataFrame containing the data.
    :param model: Model to use for trait extraction ('DSM-5' or 'ICD-11').
    :return: DataFrame containing the extracted traits.
    """
    traits = {
        'DSM-5': ['Negative Affectivity', 'Detachment', 'Antagonism', 'Disinhibition', 'Psychoticism', 'Anankastia'],
        'ICD-11': ['Negative Affectivity', 'Detachment', 'Dissociality', 'Disinhibition', 'Anankastia']
    }
    try:
        return data[traits.get(model, [])]
    except KeyError as e:
        logging.error(f"Error extracting personality traits: {e}")
        return pd.DataFrame()

def extract_icd_11_behaviour(data):
    """
    Extract ICD-11 behavior from the data.

    :param data: DataFrame containing the data.
    :return: List of extracted behaviors.
    """
    try:
        if pipelines.get('icd_11_pipeline'):
            return [pipelines['icd_11_pipeline'](text) for text in data['text_content']]
        else:
            logging.error("ICD-11 behavior extraction pipeline is not available")
            return []
    except Exception as e:
        logging.error(f"Error extracting ICD-11 behavior: {e}")
        return []

def analyze_patterns(icd_11_patterns, data):
    """
    Analyze patterns in the data using PCA and KMeans clustering.

    :param icd_11_patterns: List of ICD-11 patterns.
    :param data: DataFrame containing the data.
    :return: Tuple containing principal components and clusters.
    """
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(data['text_content'])
        
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X.toarray())
        
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(principal_components)
        
        return principal_components, clusters
    except Exception as e:
        logging.error(f"Error analyzing patterns: {e}")
        return [], []

def detect_behaviour(text, behaviour_phrases):
    """
    Detect specific behaviors in the text.

    :param text: Text to analyze.
    :param behaviour_phrases: List of behavior phrases to detect.
    :return: Boolean indicating whether any behavior phrases were detected.
    """
    try:
        return any(phrase.lower() in text.lower() for phrase in behaviour_phrases)
    except Exception as e:
        logging.error(f"Error detecting behaviour: {e}")
        return False

def perform_anomaly_detection(data):
    """
    Perform anomaly detection on the data.

    :param data: DataFrame containing the data.
    :return: DataFrame with anomaly labels.
    """
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(data['text_content'])
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(X.toarray())
        
        data['anomaly'] = anomalies
        return data
    except Exception as e:
        logging.error(f"Error performing anomaly detection: {e}")
        return data

def sentiment_trend_analysis(data):
    """
    Perform sentiment trend analysis on the data.

    :param data: DataFrame containing the data.
    """
    try:
        if pipelines.get('sentiment_pipeline'):
            data['sentiment'] = data['text_content'].apply(lambda x: pipelines['sentiment_pipeline'](x)[0]['label'])
            sentiment_counts = data['sentiment'].value_counts()
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
            plt.title('Sentiment Trend Analysis')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.show()
        else:
            logging.error("Sentiment analysis pipeline is not available")
    except Exception as e:
        logging.error(f"Error performing sentiment trend analysis: {e}")

def topic_modeling(data):
    """
    Perform topic modeling on the data.

    :param data: DataFrame containing the data.
    :return: Dictionary containing topic words.
    """
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(data['text_content'])
        
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(X)
        
        topic_words = {}
        for idx, topic in enumerate(lda.components_):
            topic_words[idx] = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]]
        
        return topic_words
    except Exception as e:
        logging.error(f"Error performing topic modeling: {e}")
        return {}

def summarize_text(text):
    """
    Summarize the given text.

    :param text: Text to summarize.
    :return: Summary of the text.
    """
    try:
        if pipelines.get('summarization_pipeline'):
            summary = pipelines['summarization_pipeline'](text, max_length=50, min_length=25, do_sample=False)
            return summary[0]['summary_text']
        else:
            logging.error("Summarization pipeline is not available")
            return text
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return text

def extract_corporate_entities(text):
    """
    Extract corporate entities (ABNs and ACNs) from the text.

    :param text: Text to analyze.
    :return: Tuple containing lists of ABNs and ACNs.
    """
    try:
        abns = ABN_PATTERN.findall(text)
        acns = ACN_PATTERN.findall(text)
        return abns, acns
    except Exception as e:
        logging.error(f"Error extracting corporate entities: {e}")
        return [], []

def perform_entity_recognition(text_data):
    """
    Perform entity recognition on the text data.

    :param text_data: List of texts to analyze.
    :return: List of recognized entities.
    """
    try:
        if pipelines.get('ner_pipeline'):
            return [pipelines['ner_pipeline'](text) for text in text_data]
        else:
            logging.error("NER pipeline is not available")
            return []
    except Exception as e:
        logging.error(f"Error performing entity recognition: {e}")
        return []

def scrape_latest_information():
    """
    Scrape the latest information from specified URLs.

    :return: List of dictionaries containing titles and contents of articles.
    """
    urls = [
        "https://example.com/latest-research",
        "https://example.com/updated-methods"
    ]
    information = []
    try:
        for url in urls:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all('article')
            for article in articles:
                title = article.find('h2').get_text()
                content = article.find('p').get_text()
                information.append({'title': title, 'content': content})
        logging.info("Scraped latest information successfully")
    except Exception as e:
        logging.error(f"Error scraping latest information: {e}")
    return information

def analyze_human_behaviour(data):
    """
    Analyze human behavior in the data.

    :param data: DataFrame containing the text data.
    :return: List of dictionaries containing analysis results.
    """
    logging.info("Starting human behaviour analysis")
    results = []
    try:
        for text in data['text_content']:
            ner_results = pipelines['icd_11_pipeline'](text) if pipelines.get('icd_11_pipeline') else []
            sentiment_results = pipelines['sentiment_pipeline'](text) if pipelines.get('sentiment_pipeline') else []
            summary = summarize_text(text)
            corporate_entities = extract_corporate_entities(text)
            entities = perform_entity_recognition([text])
            usable_quote = text if len(text.split()) < 50 else summarize_text(text)
            emotion_results = pipelines['classifier_pipeline'](text) if pipelines.get('classifier_pipeline') else []
            results.append({
                'text': text,
                'ner': ner_results,
                'sentiment': sentiment_results,
                'summary': summary,
                'corporate_entities': corporate_entities,
                'entities': entities,
                'usable_quote': usable_quote,
                'emotions': emotion_results
            })
        logging.info("Human behaviour analysis completed")
    except Exception as e:
        logging.error(f"Error analyzing human behaviour: {e}")
    return results

def ten_pass_advanced_analysis(data):
    """
    Perform ten iterations of advanced analysis on the data.

    :param data: DataFrame containing the text data.
    """
    for i in range(10):
        results = analyze_human_behaviour(data)
        data = pd.DataFrame(results)
        print(f"Iteration {i+1} completed with the following results:")
        for result in results:
            print(result)
        print(f"Updated Information: {scrape_latest_information()}")

# Example usage:
if __name__ == "__main__":
    data = pd.DataFrame({
        'text_content': [
            'The company engaged in bribery and fraud, violating several laws.',
            'You are overreacting and being dramatic. No one will believe you.',
            'The project team showed great compassion and empathy in their work.',
            'This action is illegal and unauthorized by any means.',
            'I didn’t do that, you must be remembering it wrong.'
        ]
    })
    ten_pass_advanced_analysis(data)

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
