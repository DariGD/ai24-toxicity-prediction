import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name="AppLogger", log_directory="logs", log_file="app.log"):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_path = os.path.join(log_directory, log_file)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Проверяем, чтобы не добавлять обработчик несколько раз
    if not logger.hasHandlers():
        handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger