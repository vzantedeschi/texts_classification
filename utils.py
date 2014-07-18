import logging

def get_logger(filename):
	log_format = '[%(asctime)s] %(levelname)s - %(message)s'
	log_formatter = logging.Formatter(log_format)

	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)

	handler = logging.FileHandler(filename)
	handler.setFormatter(log_formatter)

	logger.addHandler(handler)

	return logger