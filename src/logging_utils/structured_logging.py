"""
Structured logging with correlation IDs for request tracing.
"""

import logging
import json
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from contextvars import ContextVar
import sys

# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records."""

    def filter(self, record):
        record.correlation_id = correlation_id_var.get() or 'no-correlation-id'
        return True


class StructuredFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record

        Returns:
            JSON formatted log string
        """
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'correlation_id': getattr(record, 'correlation_id', 'unknown'),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


class StructuredLogger:
    """
    Structured logger with correlation ID support.
    """

    def __init__(self, name: str):
        """
        Initialize structured logger.

        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)

    def _log(
        self,
        level: int,
        message: str,
        extra_data: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ):
        """
        Log with structured data.

        Args:
            level: Log level
            message: Log message
            extra_data: Additional data to include
            exc_info: Include exception info
        """
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            "(unknown file)",
            0,
            message,
            (),
            exc_info=exc_info if exc_info else None
        )

        if extra_data:
            record.extra_data = extra_data

        self.logger.handle(record)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        exc_info = kwargs.pop('exc_info', False)
        self._log(logging.ERROR, message, kwargs, exc_info=exc_info)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        exc_info = kwargs.pop('exc_info', False)
        self._log(logging.CRITICAL, message, kwargs, exc_info=exc_info)


def setup_structured_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = True
):
    """
    Setup structured logging for the application.

    Args:
        log_level: Logging level
        log_file: Optional log file path
        json_format: Use JSON formatting
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # Add correlation ID filter
    correlation_filter = CorrelationIdFilter()
    root_logger.addFilter(correlation_filter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)

    if json_format:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(correlation_id)s - %(name)s - %(levelname)s - %(message)s'
            )
        )

    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)

        if json_format:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(correlation_id)s - %(name)s - %(levelname)s - %(message)s'
                )
            )

        root_logger.addHandler(file_handler)


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set correlation ID for current context.

    Args:
        correlation_id: Correlation ID or None to generate new one

    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """
    Get current correlation ID.

    Returns:
        Current correlation ID or None
    """
    return correlation_id_var.get()


def clear_correlation_id():
    """Clear correlation ID from current context."""
    correlation_id_var.set(None)


class CorrelationContext:
    """Context manager for correlation ID."""

    def __init__(self, correlation_id: Optional[str] = None):
        """
        Initialize correlation context.

        Args:
            correlation_id: Correlation ID or None to generate
        """
        self.correlation_id = correlation_id
        self.previous_id = None

    def __enter__(self) -> str:
        """Enter context and set correlation ID."""
        self.previous_id = get_correlation_id()
        return set_correlation_id(self.correlation_id)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous correlation ID."""
        if self.previous_id:
            set_correlation_id(self.previous_id)
        else:
            clear_correlation_id()
