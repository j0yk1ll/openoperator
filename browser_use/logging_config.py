import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def add_logging_level(level_name, level_num, method_name=None):
	"""
	Comprehensively adds a new logging level to the `logging` module and the
	currently configured logging class.

	`level_name` becomes an attribute of the `logging` module with the value
	`level_num`. `method_name` becomes a convenience method for both `logging`
	itself and the class returned by `logging.getLoggerClass()` (usually just
	`logging.Logger`). If `method_name` is not specified, `level_name.lower()` is
	used.

	To avoid accidental clobberings of existing attributes, this method will
	raise an `AttributeError` if the level name is already an attribute of the
	`logging` module or if the method name is already present

	Example
	-------
	>>> add_logging_level('TRACE', logging.DEBUG - 5)
	>>> logging.getLogger(__name__).setLevel('TRACE')
	>>> logging.getLogger(__name__).trace('that worked')
	>>> logging.trace('so did this')
	>>> logging.TRACE
	5

	"""
	if not method_name:
		method_name = level_name.lower()

	if hasattr(logging, level_name):
		raise AttributeError('{} already defined in logging module'.format(level_name))
	if hasattr(logging, method_name):
		raise AttributeError('{} already defined in logging module'.format(method_name))
	if hasattr(logging.getLoggerClass(), method_name):
		raise AttributeError('{} already defined in logger class'.format(method_name))

	# This method was inspired by the answers to Stack Overflow post
	# http://stackoverflow.com/q/2183233/2988730, especially
	# http://stackoverflow.com/a/13638084/2988730
	def log_for_level(self, message, *args, **kwargs):
		if self.is_enabled_for(level_num):
			self._log(level_num, message, args, **kwargs)

	def log_to_root(message, *args, **kwargs):
		logging.log(level_num, message, *args, **kwargs)

	logging.addLevelName(level_num, level_name)
	setattr(logging, level_name, level_num)
	setattr(logging.getLoggerClass(), method_name, log_for_level)
	setattr(logging, method_name, log_to_root)


def setup_logging():
	# Try to add RESULT level, but ignore if it already exists
	try:
		add_logging_level('RESULT', 35)  # This allows ERROR, FATAL and CRITICAL
	except AttributeError:
		pass  # Level already exists, which is fine

	log_type = os.getenv('LOG_LEVEL', 'info').lower()

	# Check if handlers are already set up
	if logging.getLogger().hasHandlers():
		return

	# Clear existing handlers
	root = logging.getLogger()
	root.handlers = []

	class BrowserUseFormatter(logging.Formatter):
		def format(self, record):
			if type(record.name) == str and record.name.startswith('browser_use.'):
				record.name = record.name.split('.')[-2]
			return super().format(record)

	# Setup single handler for all loggers
	console = logging.StreamHandler(sys.stdout)

	# additional setLevel here to filter logs
	if log_type == 'result':
		console.setLevel('RESULT')
		console.setFormatter(BrowserUseFormatter('%(message)s'))
	else:
		console.setFormatter(BrowserUseFormatter('%(levelname)-8s [%(name)s] %(message)s'))

	# Configure root logger only
	root.addHandler(console)

	# switch cases for log_type
	if log_type == 'result':
		root.setLevel('RESULT')  # string usage to avoid syntax error
	elif log_type == 'debug':
		root.setLevel(logging.DEBUG)
	else:
		root.setLevel(logging.INFO)

	# Configure browser_use logger
	browser_use_logger = logging.getLogger('browser_use')
	browser_use_logger.propagate = False  # Don't propagate to root logger
	browser_use_logger.addHandler(console)
	browser_use_logger.setLevel(root.level)  # Set same level as root logger

	logger = logging.getLogger('browser_use')
	logger.info('BrowserUse logging setup complete with level %s', log_type)

	# Check if file logging is enabled
	enable_file_logging = os.getenv('LOG_FILE', 'false').lower() == 'true'
	if enable_file_logging:
		log_file_path = os.getenv('LOG_FILE_PATH', 'browser_use.log')

		# Ensure the directory exists
		directory = os.path.dirname(log_file_path)
		if directory and not os.path.exists(directory):
			os.makedirs(directory, exist_ok=True)

		# Create file handler
		file_handler = logging.FileHandler(log_file_path)

		# Set formatter for file with timestamp
		file_formatter = logging.Formatter('%(asctime)s - %(levelname)-8s [%(name)s] %(message)s')
		file_handler.setFormatter(file_formatter)

		# Determine file log level from environment variable, default to DEBUG
		file_level_str = os.getenv('LOG_FILE_LEVEL', 'debug').lower()
		file_level = getattr(logging, file_level_str.upper(), logging.DEBUG)
		file_handler.setLevel(file_level)

		# Add file handler to root and browser_use_logger
		root.addHandler(file_handler)
		browser_use_logger.addHandler(file_handler)

		# Update root level to the minimum of current level and file_level
		new_root_level = min(root.level, file_level)
		root.setLevel(new_root_level)

		# Update browser_use_logger level to match new root level
		browser_use_logger.setLevel(new_root_level)

		logger.info('File logging enabled to %s with level %s', log_file_path, file_level_str)

	# Silence third-party loggers
	for logger_name in [
		'WDM',
		'httpx',
		'selenium',
		'playwright',
		'urllib3',
		'asyncio',
		'langchain',
		'openai',
		'httpcore',
		'charset_normalizer',
		'anthropic._base_client',
		'PIL.PngImagePlugin',
	]:
		third_party = logging.getLogger(logger_name)
		third_party.setLevel(logging.ERROR)
		third_party.propagate = False
