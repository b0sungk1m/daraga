import logging
import sys

def setup_logger():
    logger = logging.getLogger('daraga')
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File
        file_handler = logging.FileHandler('app.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

shared_logger = setup_logger()