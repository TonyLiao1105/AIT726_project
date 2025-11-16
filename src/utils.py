import os
import logging
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(log_file='app.log'):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")
    else:
        logging.info(f"Directory already exists: {directory}")