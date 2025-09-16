"""
Logging system following single responsibility principle.
"""
import logging
import sys
from typing import Optional
from .interfaces import LoggerInterface


class RAGLogger(LoggerInterface):
    """Logger implementation for the RAG system."""
    
    def __init__(self, name: str = "RAGChatApp", level: int = logging.INFO):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self._logger.info(message)
    
    def error(self, message: str, exception: Optional[Exception] = None) -> None:
        """Log error message with optional exception details."""
        if exception:
            self._logger.error(f"{message}: {str(exception)}")
        else:
            self._logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self._logger.debug(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self._logger.warning(message)
    
    def set_level(self, level: int) -> None:
        """Set logging level."""
        self._logger.setLevel(level)