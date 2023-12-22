"""Provides a logger object for the udao library."""
import logging
import sys


def _get_logger(name: str = "udao", level: int = logging.DEBUG) -> logging.Logger:
    """Generates a logger object for the UDAO library.

    Parameters
    ----------
    name : str, optional
        logger name, by default "udao".
    level : int, optional
        logging level (DEBUG, INFO...), by default logging.DEBUG

    Returns
    -------
    logging.Logger
        logger object to call for logging
    """
    _logger = logging.getLogger(name)
    _logger.setLevel(level)
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        log_format = (
            "%(asctime)s - [%(levelname)s] - %(name)s - "
            "(%(filename)s:%(lineno)d) - "
            "%(message)s"
        )
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)

        _logger.addHandler(handler)
    return _logger


logger = _get_logger()
