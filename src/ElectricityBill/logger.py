
import logging
import logging.config
import os
import sys
from datetime import datetime
from pythonjsonlogger import jsonlogger
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional
from pathlib import Path

# Constants
FORMATTER_JSON = 'json'
FORMATTER_DETAILED = 'detailed'
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT = 20
DEFAULT_LOGGER_NAME = 'ElectricityBill'


class LoggerConfigurator:
    """
    Configures and manages application logging with both file and console outputs.
    Supports JSON logging for files and detailed formatting for console output.
    """

    def __init__(
        self,
        log_dir: str = os.getenv('LOG_DIR', os.getcwd()),
        log_file_name_pattern: str = "%Y-%m-%d_%H-%M-%S",
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        formatters: Optional[Dict] = None,
        handlers: Optional[Dict] = None,
        loggers: Optional[Dict] = None,
        log_level: str = 'DEBUG',
    ):
        """
        Initialize the logger configurator.
        
        Args:
            log_dir: Directory for log files
            log_file_name_pattern: DateTime pattern for log file names
            max_bytes: Maximum size of each log file
            backup_count: Number of backup files to keep
            formatters: Custom formatters configuration
            handlers: Custom handlers configuration
            loggers: Custom loggers configuration
            log_level: Default logging level
        """
        self.log_dir = log_dir
        self.log_file_name_pattern = log_file_name_pattern
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.log_level = log_level
        
        # Initialize configurations
        self.formatters = formatters if formatters else self._default_formatters()
        self.handlers = handlers if handlers else self._default_handlers()
        self.loggers = loggers if loggers else self._default_loggers()
        
        self._configure_logging()

    def _default_formatters(self) -> Dict[str, Dict[str, Any]]:
        """Define default formatters for different logging outputs."""
        return {
            'standard': {
                'format': '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
            },
            FORMATTER_DETAILED: {
                'format': '[%(asctime)s] [%(levelname)s] %(name)s - %(module)s:%(lineno)d - %(message)s'
            },
            FORMATTER_JSON: {
                '()': jsonlogger.JsonFormatter,
                'format': '%(asctime)s %(name)s %(levelname)s %(module)s %(lineno)d %(message)s'
            },
        }

    def _default_handlers(self) -> Dict[str, Dict[str, Any]]:
        """Define default handlers for file and console logging."""
        log_file_path = self._get_log_file_path()
        return {
            'file': {
                'level': self.log_level,
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': log_file_path,
                'maxBytes': self.max_bytes,
                'backupCount': self.backup_count,
                'encoding': 'utf8',
                'formatter': FORMATTER_JSON
            },
            'console': {
                'level': self.log_level,
                'class': 'logging.StreamHandler',
                'formatter': FORMATTER_DETAILED,
                'stream': sys.stdout
            }
        }

    def _default_loggers(self) -> Dict[str, Dict[str, Any]]:
        """Define default logger configuration."""
        return {
            DEFAULT_LOGGER_NAME: {
                'handlers': ['file', 'console'],
                'level': self.log_level,
                'propagate': False
            }
        }

    def _get_log_file_path(self) -> str:
        """Generate and ensure the log file path exists."""
        log_file_name = f"{datetime.now().strftime(self.log_file_name_pattern)}.log"
        log_path = Path(self.log_dir) / 'logs' / log_file_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return str(log_path)

    def _configure_logging(self) -> None:
        """Apply the logging configuration."""
        try:
            logging_config = {
                'version': 1,
                'disable_existing_loggers': False,
                'formatters': self.formatters,
                'handlers': self.handlers,
                'loggers': self.loggers,
            }
            logging.config.dictConfig(logging_config)
        except Exception as e:
            print(f"Failed to configure logging: {e}")
            raise
            
    def get_logger(self, name:str = DEFAULT_LOGGER_NAME) -> logging.Logger:
       """
        Get a configured logger instance.
        
        Args:
            name: Name of the logger to retrieve
            
        Returns:
            logging.Logger: Configured logger instance
        """
       return logging.getLogger(name)


# Create and configure the logger (once)
logger_configurator = LoggerConfigurator()
logger = logger_configurator.get_logger()


if __name__ == "__main__":
    # Example usage with error handling
    try:
       
        # Test all logging levels
        logger.debug("Debug message with extra context", extra={"user_id": 123})
        logger.info("Info message about system state", extra={"status": "running"})
        logger.warning("Warning about resource usage", extra={"cpu": "85%"})
        logger.error("Error in process", extra={"error_code": 500})
        logger.critical("Critical system error", extra={"shutdown": True})
        
    except Exception as e:
        print(f"Failed to initialize logging: {e}")
        sys.exit(1)