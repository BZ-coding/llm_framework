import logging

import datasets
import transformers

log_level = logging.INFO
logger = logging.getLogger("__main__")
console_handler = logging.StreamHandler()

# Setup logging
logger.setLevel(log_level)
console_handler.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()


def get_logger(log_file=None):
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
        transformers.utils.logging.add_handler(file_handler)
    return logger
