# src/factory/logger_factory.py

import logging
import os


class LoggerFactory:
    @staticmethod
    def create_logger(log_dir: str, log_filename: str = 'train.log') -> logging.Logger:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)

        # Create logger
        logger = logging.getLogger("train_logger")
        logger.setLevel(logging.INFO)

        # Avoid adding duplicate handlers if called multiple times
        if not logger.handlers:
            # File handler
            file_handler = logging.FileHandler(log_path, mode='a')
            file_handler.setFormatter(logging.Formatter(
                fmt='%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                fmt='%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))

            # Add both handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger
