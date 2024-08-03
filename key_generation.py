# E:\Public_Interest_Project\Modules_Scripts\key_generation.py

import subprocess
import os

def generate_key(config_file, output_dir, key_name):
    try:
        os.makedirs(output_dir, exist_ok=True)
        key_path = os.path.join(output_dir, key_name)
        
        # Run OpenSSL command to generate key
        command = f'openssl req -new -newkey rsa:2048 -days 365 -nodes -x509 -keyout {key_path}.key -out {key_path}.crt -config {config_file}'
        subprocess.run(command, shell=True, check=True)
        return True, f"Key generated at {key_path}.key"
    except subprocess.CalledProcessError as e:
        return False, f"Error generating key: {e}"

# Example usage:
# generate_key("E:\\Public_Interest_Project\\config_files\\openssl_automated_user_responses.txt", "E:\\Public_Interest_Project\\Keys", "user_key")
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
