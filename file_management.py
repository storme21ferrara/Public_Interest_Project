import os
import logging
import json
import shutil
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO)

def load_config(config_path='file_management_config.json'):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def load_terminal_config(config_path='E:/Public_interest_project/config_files/terminal_config.json'):
    with open(config_path, 'r') as config_file:
        terminal_config = json.load(config_file)
    return terminal_config

def delete_old_files(directory, days_old):
    now = time.time()
    cutoff = now - (days_old * 86400)
    files_deleted = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_stat = os.stat(file_path)
            if file_stat.st_mtime < cutoff:
                os.remove(file_path)
                files_deleted.append(filename)
                logging.info(f"Deleted old file: {filename}")
    return files_deleted

def compress_large_files(directory, size_threshold):
    files_compressed = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and os.path.getsize(file_path) > size_threshold:
            shutil.make_archive(file_path, 'zip', file_path)
            files_compressed.append(filename)
            logging.info(f"Compressed large file: {filename}")
    return files_compressed

def split_large_files(directory, size_threshold, part_size):
    files_split = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and os.path.getsize(file_path) > size_threshold:
            part_num = 1
            with open(file_path, 'rb') as f:
                while chunk := f.read(part_size):
                    part_filename = f"{filename}.part{part_num}"
                    part_path = os.path.join(directory, part_filename)
                    with open(part_path, 'wb') as part_file:
                        part_file.write(chunk)
                    part_num += 1
            os.remove(file_path)
            files_split.append(filename)
            logging.info(f"Split large file: {filename} into {part_num-1} parts")
    return files_split

def fetch_web_data(url, params):
    response = requests.get(url, params=params)
    if response.status_code == 200:
        logging.info("Fetched web data successfully.")
        return response.json()
    else:
        logging.error(f"Failed to fetch web data. Status code: {response.status_code}")
        return None

def query_database(connection_string, query):
    engine = create_engine(connection_string)
    with engine.connect() as connection:
        result = connection.execute(query)
        data = pd.DataFrame(result.fetchall(), columns=result.keys())
        logging.info("Queried database successfully.")
    return data

def manage_files(config, terminal_config):
    with ThreadPoolExecutor() as executor:
        old_files_future = executor.submit(delete_old_files, config['project_directory'], config['delete_days'])
        compressed_files_future = executor.submit(compress_large_files, config['project_directory'], config['compress_size_threshold'])
        split_files_future = executor.submit(split_large_files, config['project_directory'], config['split_size_threshold'], config['split_part_size'])
        web_data_future = executor.submit(fetch_web_data, config['web_data_url'], config['web_data_params'])
        db_query_future = executor.submit(query_database, config['db_connection_string'], config['db_query'])

        old_files = old_files_future.result()
        compressed_files = compressed_files_future.result()
        split_files = split_files_future.result()
        web_data = web_data_future.result()
        db_data = db_query_future.result()

        logging.info(f"Old files deleted: {old_files}")
        logging.info(f"Files compressed: {compressed_files}")
        logging.info(f"Files split: {split_files}")
        logging.info(f"Web data: {web_data}")
        logging.info(f"Database data: {db_data}")

        optimize_system(terminal_config)

def optimize_system(terminal_config):
    logging.info(f"Optimizing system: {terminal_config['system']['name']} with {terminal_config['system']['ram']} RAM")
    # Additional optimization logic can be added here based on terminal_config

def main():
    config = load_config()
    terminal_config = load_terminal_config()
    manage_files(config, terminal_config)

if __name__ == "__main__":
    main()
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
