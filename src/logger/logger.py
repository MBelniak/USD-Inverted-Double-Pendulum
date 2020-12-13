import logging
import sys


def get_logger(name: str):
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    handler = logging.FileHandler(f'logger/{name}.log')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger


a3c_logger = get_logger("A3C")
