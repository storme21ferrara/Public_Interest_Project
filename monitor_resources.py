import psutil
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def monitor_resources():
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        logging.info(f'CPU Usage: {cpu_usage}%')
        logging.info(f'Memory Usage: {memory_usage}%')
        time.sleep(60)

if __name__ == '__main__':
    monitor_resources()
