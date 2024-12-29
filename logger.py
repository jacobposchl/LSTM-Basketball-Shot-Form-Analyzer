# logger.py

import logging

def setup_logging():
    """
    Configures the logging settings for the application.

    - Logs messages with level INFO and above.
    - Formats logs with timestamp, log level, and message.
    - Outputs logs to the console.
    - Optionally, logs can be written to a file by uncommenting the FileHandler.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create console handler and set level to INFO
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(ch)

    # Optional: Add FileHandler to log messages to a file
    # fh = logging.FileHandler('project_marissa.log')
    # fh.setLevel(logging.INFO)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
