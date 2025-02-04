
import sys
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass

# Initialize the logger
logger = logging.getLogger('LeadScoring')
logging.basicConfig(level=logging.ERROR)

@dataclass(frozen=True)
class ErrorDetails:
    """
    A dataclass to hold error details.
    
    Attributes:
        exc_type (type): The type of the exception.
        exc_value (BaseException): The exception instance.
        exc_traceback (type): The traceback object.
    """
    exc_type: type
    exc_value: BaseException
    exc_traceback: type

class CustomException(Exception):
    """
    Custom exception class for the Lead Scoring application.

    Attributes:
        message (str): A detailed, formatted error message
        error (Exception): The original exception instance.
        error_details (ErrorDetails): An instance of ErrorDetails.
        context (Optional[Dict]): Optional additional context data.
        error_type (str): A string indicating the type of error
    """
    def __init__(
        self, 
        error: Exception, 
        error_type: Optional[str] = None, 
        context: Optional[Dict] = None,
        log_immediately: bool = False
    ):
        self.error = error
        self.context = context or {}  # Initialize as empty dict if None
        self.error_type = error_type if error_type else type(error).__name__

        # Capture traceback details
        error_details = sys.exc_info()
        self.error_details = ErrorDetails(*error_details)

        # Format the error message
        self.message = self._format_error_message()
        
        super().__init__(self.message)

        # Optionally log the error immediately
        if log_immediately:
            self.log_error()
    
    def _format_error_message(self) -> str:
        """
        Format detailed error information.
        
        Returns:
            str: Formatted error message with file, line number, and context details.
        """
        try:
            file_name = self.error_details.exc_traceback.tb_frame.f_code.co_filename
            line_number = self.error_details.exc_traceback.tb_lineno

            formatted_error_message = (
                f"Error Type: [{self.error_type}]\n"
                f"File: {file_name}\n"
                f"Line Number: [{line_number}]\n"
                f"Error message: [{self.error}]"
            )

            if self.context:
                context_str = '\n'.join(f"  {k}: {v}" for k, v in self.context.items())
                formatted_error_message += f"\nContext:\n{context_str}"

            return formatted_error_message
            
        except AttributeError:
            return f"Error: {self.error} (Stack trace details unavailable)"

    def log_error(self, level: int = logging.ERROR) -> None:
        """
        Log the formatted error message at the specified level.
        
        Args:
            level (int): The logging level to use (default: logging.ERROR)
        """
        logger.log(level, self.message)

    def __str__(self) -> str:
        """Return the formatted error message."""
        return self.message

def example_function():
    try:
        # Simulating an error
        1 / 0
    except Exception as e:
        raise CustomException(
            e, 
            error_type="Database",  
            context={"operation": "division", "status": "failed"}
        )

if __name__ == "__main__":
    try:
        logger.info("Starting the example function.")
        example_function()
    except CustomException as e:
        # The error is already logged by default, but you can log it again if needed
        logger.error(f"Additional error logging: {e}")