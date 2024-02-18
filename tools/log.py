import logging

import datasets
import transformers

transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()


def get_logger(log_level, logger_log_level=None, logger=None):
    if logger_log_level is None:
        logger_log_level = log_level
    if logger is None:
        logger = logging.getLogger("__main__")
    console_handler = logging.StreamHandler()
    logger.setLevel(logger_log_level)
    console_handler.setLevel(logger_log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    return logger
