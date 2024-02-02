import logging
import sys


class ConsoleFormatter(logging.Formatter):
    """ https://stackoverflow.com/a/56944256 """
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    
    
console_formatter = ConsoleFormatter()
file_formatter = logging.Formatter(fmt="%(levelname)s - %(message)s (%(filename)s:%(lineno)d)")

# file_handler = logging.FileHandler(logging_path)
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(console_formatter)

ipc_logger = logging.getLogger('sapienipc')
ipc_logger.setLevel(logging.INFO)
# logger.addHandler(file_handler)
ipc_logger.handlers.clear()
ipc_logger.propagate = False
ipc_logger.addHandler(console_handler)
