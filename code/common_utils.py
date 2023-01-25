import logging
import random
import datetime
import numpy as np
import torch
import os


BASE_LOG_PATH = "/service/logs/"
BASE_LOG_NAME = "PATENT_log"

if not os.path.exists(BASE_LOG_PATH):
    os.mkdir(BASE_LOG_PATH)
    
logger = logging.getLogger("PATENT")

def setup_logger(logger):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_formatter = logging.Formatter("[PATENT] %(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(log_formatter)
    logger.addHandler(console)
    timedfilehandler = logging.handlers.TimedRotatingFileHandler(filename=os.path.join(BASE_LOG_PATH, BASE_LOG_NAME), when='d', interval=1, encoding='utf-8')
    timedfilehandler.setFormatter(log_formatter)
    timedfilehandler.suffix = "%Y-%m-%d"
    logger.addHandler(timedfilehandler)