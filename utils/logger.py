import logging
import os

def get_logger(name, log_file='logs/application.log', level=logging.INFO):
    """
    Create and Configure a Logger
    
    Args:
        name (str): Name of the logger (typically __name__ of the calling module).
        log_file (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Create a console handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Define a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.handlers:  # Avoid duplicate handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
