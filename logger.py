# logger.py

import logging
import os
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("study_buddy_logger")
logger.setLevel(logging.DEBUG)  

if not os.path.exists("logs"):
    os.makedirs("logs")

file_handler = RotatingFileHandler("logs/study_buddy.log", maxBytes=5*1024*1024, backupCount=2)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)