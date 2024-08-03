import subprocess
import os
import logging

def query_icd_api(icd_code):
    api_executable = r"C:\Program Files\ICD-API\ICD-API.exe"
    client_id = "a42ca42f-fe84-4134-839a-78f0d7754cba_aaca6aba-e80b-4fc1-91c5-6ddd0056b906"
    client_secret = "24GdB5PAgQ0gi3JG6qtx7nxqWjZpK51eEkuuWmj26YM"

    # Ensure the ICD code is valid (simple validation, can be extended)
    if not icd_code:
        raise ValueError("ICD code must be provided")

    # Set environment variable
    env = os.environ.copy()
    env['acceptLicense'] = 'true'

    try:
        # Log the API call
        logging.info(f"Calling ICD API with code: {icd_code}")

        # Call the API executable
        result = subprocess.run([api_executable, '--client_id', client_id, '--client_secret', client_secret, '--code', icd_code],
                                capture_output=True, text=True, env=env)
        
        # Check for errors
        if result.returncode != 0:
            logging.error(f"API call failed: {result.stderr}")
            raise RuntimeError(f"API call failed: {result.stderr}")

        # Return the result
        return result.stdout

    except Exception as e:
        logging.error(f"Error calling ICD API: {e}")
        raise

# Example usage
if __name__ == "__main__":
    icd_code = "ICD-11-06-6a05" "Attention Deficit Hyperactivity Disorder"
    try:
        response = query_icd_api(icd_code)
        print(response)
    except Exception as e:
        print(f"Failed to query ICD API: {e}")
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
