import logging

def get_logger():
	log_format = '[%(asctime)s] %(levelname)s - %(message)s'
	log_formatter = logging.Formatter(log_format)

	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)

	handler = logging.FileHandler("./logs/classification.log")
	handler.setFormatter(log_formatter)

	logger.addHandler(handler)

	return logger