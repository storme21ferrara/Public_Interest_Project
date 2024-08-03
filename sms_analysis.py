import os
import pandas as pd
import glob
import logging
import re
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from multiprocessing import Pool, cpu_count
from textblob import TextBlob
from geopy.geocoders import Nominatim
import spacy
import stanza
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from transformers import pipeline

# Configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Directory paths
input_directory = "E:\\Public_interest_Project\\Data\\Input"
output_dataframe_directory = "A:\\Public_interest_Project_Phoenix\\Output\\Data_Frames"
output_database_directory = "A:\\Public_interest_Project_Phoenix\\Output\\Data_Bases"

# Columns for the final dataframe
columns = [
    "Message Number", "Title", "Message Content", "Date And Time", "Attachments",
    "Name", "Phone", "Sent/Received", "Message Type", "Date", "Time", "Delivery Report", 
    "MMS Link", "MMS Subject", "Tone", "Intentions", "Malice", "Gaslighting", 
    "Attack Behaviour", "Externalisation", "Jurisdiction", "Location", 
    "Entity", "Entity Type"
]

# Initialize geolocator
geolocator = Nominatim(user_agent="sms_analysis")

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# Initialize Stanza
stanza.download('en')
nlp_stanza = stanza.Pipeline('en')

# Initialize Transformers
ner = pipeline("ner")

def extract_named_entities(text):
    entities = []
    for ent in nlp(text).ents:
        entities.append((ent.text, ent.label_))
    return entities

def nltk_extract_named_entities(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    entities = []
    for chunk in chunked:
        if type(chunk) == Tree:
            entity = " ".join([c[0] for c in chunk])
            entity_type = chunk.label()
            entities.append((entity, entity_type))
    return entities

def stanza_extract_named_entities(text):
    doc = nlp_stanza(text)
    entities = [(ent.text, ent.type) for ent in doc.ents]
    return entities

def transformers_extract_named_entities(text):
    entities = ner(text)
    return [(entity['word'], entity['entity']) for entity in entities]

def analyze_text(content):
    analysis = TextBlob(content)
    tone = analysis.sentiment.polarity
    intentions = "Positive" if tone > 0 else "Negative" if tone < 0 else "Neutral"
    malice = "Present" if "threat" in content.lower() or "harm" in content.lower() else "Absent"
    gaslighting = "Present" if "crazy" in content.lower() or "imagining things" in content.lower() else "Absent"
    attack_behaviour = "Present" if "attack" in content.lower() or "hurt" in content.lower() else "Absent"
    externalisation = "Present" if "your fault" in content.lower() else "Absent"
    
    return tone, intentions, malice, gaslighting, attack_behaviour, externalisation

def extract_event_details(content):
    event_date = re.search(r'\b\d{4}-\d{2}-\d{2}\b', content)
    event_time = re.search(r'\b\d{2}:\d{2}\b', content)
    event_summary = content[:100] + "..." if len(content) > 100 else content
    event_date = event_date.group(0) if event_date else ""
    event_time = event_time.group(0) if event_time else ""
    
    location = None
    locations = geolocator.geocode(content, exactly_one=False, timeout=10)
    if locations:
        location = locations[0].address

    return event_date, event_time, event_summary, location

def process_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if set(columns).issubset(df.columns):
            logging.info(f"Processing CSV file: {file_path}")
            with Pool(cpu_count()) as pool:
                results = pool.map(process_message_row, df.iterrows())
            df = pd.DataFrame(results, columns=columns)
            return df
        else:
            logging.warning(f"CSV file {file_path} does not have the required columns.")
            return pd.DataFrame(columns=columns)
    except Exception as e:
        logging.error(f"Error processing CSV file {file_path}: {e}")
        return pd.DataFrame(columns=columns)

def process_message_row(row):
    tone, intentions, malice, gaslighting, attack_behaviour, externalisation = analyze_text(row[1]["Message Content"])
    event_date, event_time, event_summary, location = extract_event_details(row[1]["Message Content"])
    return {
        "Message Number": row[1]["Message Number"],
        "Title": row[1]["Title"],
        "Message Content": row[1]["Message Content"],
        "Date And Time": row[1]["Date And Time"],
        "Attachments": row[1]["Attachments"],
        "Name": row[1]["Name"],
        "Phone": row[1]["Phone"],
        "Sent/Received": row[1]["Sent/Received"],
        "Message Type": row[1]["Message Type"],
        "Date": row[1]["Date"],
        "Time": row[1]["Time"],
        "Delivery Report": row[1]["Delivery Report"],
        "MMS Link": row[1]["MMS Link"],
        "MMS Subject": row[1]["MMS Subject"],
        "Tone": tone,
        "Intentions": intentions,
        "Malice": malice,
        "Gaslighting": gaslighting,
        "Attack Behaviour": attack_behaviour,
        "Externalisation": externalisation,
        "Jurisdiction": "Australia",
        "Location": location,
        "Entity": None,
        "Entity Type": None
    }

def process_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        messages = re.findall(
            r"Message Number: (\d+).*?Title: (.*?)\nMessage Content: (.*?)\nDate And Time: (.*?)\nAttachments: (.*?)\nName: (.*?)\nPhone: (.*?)\nSent/Received: (.*?)\nMessage Type: (.*?)\nDate: (.*?)\nTime: (.*?)\nDelivery Report: (.*?)\nMMS Link: (.*?)\nMMS Subject: (.*?)\n", 
            content, re.DOTALL
        )

        data = []
        for message in messages:
            tone, intentions, malice, gaslighting, attack_behaviour, externalisation = analyze_text(message[2])
            event_date, event_time, event_summary, location = extract_event_details(message[2])
            data.append({
                "Message Number": message[0],
                "Title": message[1],
                "Message Content": message[2],
                "Date And Time": message[3],
                "Attachments": message[4],
                "Name": message[5],
                "Phone": message[6],
                "Sent/Received": message[7],
                "Message Type": message[8],
                "Date": message[9],
                "Time": message[10],
                "Delivery Report": message[11],
                "MMS Link": message[12],
                "MMS Subject": message[13],
                "Tone": tone,
                "Intentions": intentions,
                "Malice": malice,
                "Gaslighting": gaslighting,
                "Attack Behaviour": attack_behaviour,
                "Externalisation": externalisation,
                "Jurisdiction": "Australia",
                "Location": location,
                "Entity": None,
                "Entity Type": None
            })
        
        df = pd.DataFrame(data, columns=columns)
        logging.info(f"Processing TXT file: {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error processing TXT file {file_path}: {e}")
        return pd.DataFrame(columns=columns)

def process_file(file_path):
    if file_path.endswith(".csv"):
        return process_csv_file(file_path)
    elif file_path.endswith(".txt"):
        return process_txt_file(file_path)
    else:
        logging.warning(f"Unsupported file format: {file_path}")
        return pd.DataFrame(columns=columns)

def save_data_to_dataframe(df):
    try:
        output_file = os.path.join(output_dataframe_directory, "sms_data.csv")
        df.to_csv(output_file, index=False)
        logging.info(f"Data frame saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving data frame: {e}")

def save_data_to_database(df):
    try:
        database_file = os.path.join(output_database_directory, "sms_data.db")
        engine = create_engine(f"sqlite:///{database_file}")
        df.to_sql('sms_data', con=engine, if_exists='replace', index=False)
        logging.info(f"Data saved to database at {database_file}")
    except Exception as e:
        logging.error(f"Error saving data to database: {e}")

def build_combined_csv():
    try:
        logging.info("Starting the combination of CSV and TXT files...")
        all_files = glob.glob(os.path.join(input_directory, "*"))
        
        with Pool(cpu_count()) as pool:
            results = pool.map(process_file, all_files)
        
        combined_df = pd.concat(results, ignore_index=True)
        save_data_to_dataframe(combined_df)
        save_data_to_database(combined_df)
        logging.info("Combined CSV file created and data saved to database")
    except Exception as e:
        logging.error(f"Error in building combined CSV: {e}")

def schedule_regular_scans():
    scheduler = BackgroundScheduler()
    scheduler.add_job(build_combined_csv, 'interval', minutes=60)
    scheduler.start()
    logging.info("Scheduled regular scans every 60 minutes.")

if __name__ == "__main__":
    schedule_regular_scans()
    try:
        while True:
            build_combined_csv()
            time.sleep(3600)  # Perform the task every hour
    except KeyboardInterrupt:
        logging.info("Stopping the script.")
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
