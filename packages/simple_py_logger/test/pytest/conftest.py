"""
Used for creating fixtures for pytest
"""

import logging
import logging.handlers as log_handlers
import socket

import pytest

import logger


# For main Logger class
@pytest.fixture(scope="class")
def logger_class():
    yield logger.Logger("TESTER")


@pytest.fixture(scope="class")
def logger_class_error():
    yield logger.Logger("ERROR_TESTER", "/fake/path", "error_")


# For handler helpers
@pytest.fixture(scope="class")
def base_log_level(request):
    if request.cls.__name__ == "TestHandlerFailures":
        yield "10"
    else:
        yield logger._std_log_level("debug")


@pytest.fixture(scope="module")
def base_formatter():
    yield logging.Formatter("%(name)s - %(message)s")


@pytest.fixture(scope="class")
def handler_params(base_log_level, base_formatter):
    yield {
        "log_level_int": base_log_level,
        "formatter": base_formatter,
    }


@pytest.fixture(scope="function")
def rich_params(request):
    if request.cls.__name__ == "TestHandlerFailures":
        yield {
            "markup": [],
            "rich_tracebacks": {},
        }
    else:
        yield {
            "markup": True,
            "rich_tracebacks": False,
        }


@pytest.fixture(scope="function")
def file_params(request):
    if request.cls.__name__ == "TestHandlerFailures":
        yield {
            "log_path": 1000,
            "mode": -123,
            "encoding": -100000,
            "delay": "no?",
            "errors": -123,
        }
    else:
        yield {
            "log_path": logger._derive_log_path(None),
            "mode": "a",
            "encoding": "utf-8",
            "delay": False,
            "errors": None,
        }


@pytest.fixture(scope="function")
def rotate_file_params(request):
    if request.cls.__name__ == "TestHandlerFailures":
        yield {
            "log_path": 1000,
            "mode": "a",
            "maxBytes": -123,
            "backupCount": -100000,
            "encoding": "utf-8",
            "delay": False,
            "errors": None,
        }
    else:
        yield {
            "log_path": logger._derive_log_path(None),
            "mode": "a",
            "maxBytes": 100,
            "backupCount": 10,
            "encoding": "utf-8",
            "delay": False,
            "errors": None,
        }


@pytest.fixture(scope="function")
def rotate_time_params(request):
    if request.cls.__name__ == "TestHandlerFailures":
        yield {
            "log_path": 1000,
            "when": "h",
            "interval": -123,
            "backupCount": -100000,
            "encoding": "utf-8",
            "delay": False,
            "utc": False,
            "atTime": None,
            "errors": None,
        }
    else:
        yield {
            "log_path": logger._derive_log_path(None),
            "when": "h",
            "interval": 1,
            "backupCount": 10,
            "encoding": "utf-8",
            "delay": False,
            "utc": False,
            "atTime": None,
            "errors": None,
        }


@pytest.fixture(scope="function")
def syslog_params(request):
    if request.cls.__name__ == "TestHandlerFailures":
        yield {
            "address": 1000,
            "facility": -123,
            "socktype": -100000,
        }
    else:
        yield {
            "address": ("localhost", log_handlers.SYSLOG_UDP_PORT),
            "facility": log_handlers.SysLogHandler.LOG_USER,
            "socktype": socket.SOCK_DGRAM,
        }


@pytest.fixture(scope="function")
def http_params(request):
    if request.cls.__name__ == "TestHandlerFailures":
        yield {
            "host": 1000,
            "url": -123,
            "method": "GET",
            "secure": False,
            "credentials": None,
            "context": None,
        }
    else:
        yield {
            "host": "localhost",
            "url": "http://localhost",
            "method": "GET",
            "secure": False,
            "credentials": None,
            "context": None,
        }
