import logging
import sys

# Only one console_handler - otherwise it prints messages multiple times if there are multiple loggers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

def get_logger(name: str):
    logger = logging.getLogger(__name__)

    handler = logging.FileHandler(f'logger/{name}.log')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


a3c_logger = get_logger("A3C")
dqn_logger = get_logger("DDQN")
