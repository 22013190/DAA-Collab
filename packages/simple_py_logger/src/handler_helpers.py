"""
This is a set of functions that to aid in creation of handlers

TODO:
    [ ] Add Pydantic for data validation.
"""

import datetime
import logging
import logging.handlers as log_handlers
import socket
import ssl
import sys
from typing import Literal, Tuple

from rich.logging import RichHandler


def create_console_handler(
    log_level_int: int,
    formatter: logging.Formatter,
    stream=sys.stdout,
) -> logging.StreamHandler:
    """
    Creates basic console handler for logging.

    Args:
        log_level_int:  Integer value for logging level.
        formatter    :  A formatter instance to format message for handler.
        stream       :  Streaming output such as TextI/O etc. Usually standard
                        sys.stdout or sys.stderr is enough. Custom stream
                        instances can be passed as well.

    Returns:
        logging.StreamHandler: A stream handler instance to attach to logger.
    """
    console_h = logging.StreamHandler(stream)
    console_h.setFormatter(formatter)
    console_h.setLevel(log_level_int)
    return console_h


def create_rich_console_handler(
    log_level_int: int,
    formatter: logging.Formatter,
    markup: bool = True,
    rich_tracebacks: bool = True,
) -> RichHandler:
    """
    Create a rich handler for console logging.

    Args:
        log_level_int  :    Integer value for logging level.
        formatter      :    A formatter instance to format message for handler.
        markup         :    Set to enable markup formatting.
        rich_tracebacks:    Set to output more details during tracebacks.

    Returns:
        RichHandler    :    A rich handler instance to attach to logger.
    """
    rich_console_h = RichHandler(
        level=log_level_int,
        markup=markup,
        rich_tracebacks=rich_tracebacks,
    )
    rich_console_h.setFormatter(formatter)
    rich_console_h.setLevel(log_level_int)
    return rich_console_h


def create_file_handler(
    log_level_int: int,
    formatter: logging.Formatter,
    log_path: str,
    mode: str = "a",
    encoding: str | None = None,
    delay: bool = False,
    errors: str | None = None,
) -> logging.FileHandler:
    """
    Creates basic file handler for logging.

    Args:
        log_level_int     : Integer value for logging level.
        formatter         : A formatter instance to format message for handler.
        log_path          : Path to desired file for output. Try to be absolute path
                            for consistency and accuracy.
        mode              : Value to open file as specified mode. Refer
                            to: https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
        encoding          : String to open file with specified encoding otherwise
                            None as default.
        delay             : Boolean to defer file opening until first call to `emit()`.
        errors            : String variable to determines how errors are handled otherwise
                            None as default.

    Returns:
        logging.FileHandler: A file handler instance to attach to logger.
    """
    file_h = logging.FileHandler(
        filename=log_path,
        mode=mode,
        encoding=encoding,
        delay=delay,
        errors=errors,
    )
    file_h.setFormatter(formatter)
    file_h.setLevel(log_level_int)
    return file_h


def create_rotate_file_handler(
    log_level_int: int,
    formatter: logging.Formatter,
    log_path: str,
    mode: str = "a",
    maxBytes: int = 100 * 1024,  # Set to 100mb by default
    backupCount: int = 1,
    encoding: str | None = None,
    delay: bool = False,
    errors: str | None = None,
) -> log_handlers.RotatingFileHandler:
    """
    Creates file rotation handler based on time for logging.

    Further options
    refer to: https://docs.python.org/3/library/logging.handlers.html#rotatingfilehandler

    Args:
        log_level_int : Integer value for logging level.
        formatter     : A formatter instance to format message for handler.
        log_path      : Path to desired file for output. Try to be absolute path
                        for consistency and accuracy.
        mode          : Value to open file as specified mode. Refer
                        to: https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
        maxBytes      : Number of bytes to store per file before rollover.
        backupCount   : Determines number of similar files are kept as backup
                        based on specified time. Interval variable will affect
                        the categorisation of files.
        encoding      : String to open file with specified encoding otherwise
                        None as default.
        delay         : Boolean to defer file opening until first call to `emit()`.
        errors        : String variable to determines how errors are handled otherwise
                        None as default.

    Returns:
        logging.handlers.RotatingFileHandler:   A byte based rotating file
                                                handler to attach to logger.
    """
    rot_byte_h = log_handlers.RotatingFileHandler(
        filename=log_path,
        mode=mode,
        maxBytes=maxBytes,
        backupCount=backupCount,
        encoding=encoding,
        delay=delay,
        errors=errors,
    )
    rot_byte_h.setFormatter(formatter)
    rot_byte_h.setLevel(log_level_int)
    return rot_byte_h


def create_rotate_time_handler(
    log_level_int: int,
    formatter: logging.Formatter,
    log_path: str,
    when: str = "h",
    interval: int = 1,
    backupCount: int = 0,
    encoding: str | None = None,
    delay: bool = False,
    utc: bool = False,
    atTime: datetime.time | None = None,
    errors: str | None = None,
) -> log_handlers.TimedRotatingFileHandler:
    """
    Creates file rotation handler based on time for logging.

    Further options
    refer to: https://docs.python.org/3/library/logging.handlers.html#timedrotatingfilehandler

    Args:
        log_level_int : Integer value for logging level.
        formatter     : A formatter instance to format message for handler.
        log_path      : Path to desired file for output. Try to be absolute path
                        for consistency and accuracy.
        when          : See reference above for allowed string values.
        interval      : Works with `when` value to determine next rollover
                        period of file.
        backupCount   : Determines number of similar files are kept as backup
                        based on specified time. Interval variable will affect
                        the categorisation of files.
        encoding      : String to open file with specified encoding otherwise
                        None as default.
        delay         : Boolean to defer file opening until first call to `emit()`.
        utc           : Boolean to determine if UTC time will be used otherwise
                        it is local time.
        atTime        : A datetime.time instance otherwise None as default.
                        Specifies when the rollover occurs.
        errors        : String variable to determines how errors are handled otherwise
                        None as default.

    Returns:
        logging.handlers.TimedRotatingFileHandler:  A time based rotating file
                                                    handler to attach to logger.
    """
    rot_time_h = log_handlers.TimedRotatingFileHandler(
        filename=log_path,
        when=when,
        interval=interval,
        backupCount=backupCount,
        encoding=encoding,
        delay=delay,
        utc=utc,
        atTime=atTime,
        errors=errors,
    )
    rot_time_h.setFormatter(formatter)
    rot_time_h.setLevel(log_level_int)
    return rot_time_h


def create_syslog_handler(
    log_level_int: int,
    formatter: logging.Formatter,
    address: Tuple[str, int] = ("localhost", log_handlers.SYSLOG_UDP_PORT),
    facility: int = log_handlers.SysLogHandler.LOG_USER,
    socktype: socket.SocketKind = socket.SOCK_DGRAM,
) -> log_handlers.SysLogHandler:
    """
    Creates syslog handler for logging.

    Args:
        log_level_int : Integer value for logging level.
        formatter     : A formatter instance to format message for handler.
        address       : A tuple of IP string and int port number for connection.
        facility      : An int number to Syslog facility levels.
        socktype      : An int number to socket type used for connection.

    Returns:
        logging.handlers.SysLogHandler:   A syslog file handler to attach to logger.
    """
    syslog_h = log_handlers.SysLogHandler(
        address=address,
        facility=facility,
        socktype=socktype,
    )
    syslog_h.setFormatter(formatter)
    syslog_h.setLevel(log_level_int)
    return syslog_h


def create_http_handler(
    log_level_int: int,
    formatter: logging.Formatter,
    host: str,
    url: str,
    method: Literal["GET", "POST"] = "GET",
    secure: bool = False,
    credentials: Tuple[str, str] | None = None,
    context: ssl.SSLContext | None = None,
) -> log_handlers.HTTPHandler:
    """
    Creates HTTP handler for logging.

    Args:
        log_level_int : Integer value for logging level.
        formatter     : A formatter instance to format message for handler.
        host          : String value that can be formatted as `host:port`.
        url           : String value of request URL.
        secure        : True for HTTPS or False for HTTP connection.
        credentials   : A tuple of (username and password) to be placed in HTTP
                        Authorization Header using Basic Authentication.
        context       : SSLContext instance for HTTPS configuration otherwise None
                        as default.

    Returns:
        logging.handlers.HTTPHandler:   A syslog file handler to attach to logger.
    """
    http_h = log_handlers.HTTPHandler(
        host=host,
        url=url,
        method=method,
        secure=secure,
        credentials=credentials,
        context=context,
    )
    http_h.setFormatter(formatter)
    http_h.setLevel(log_level_int)
    return http_h
