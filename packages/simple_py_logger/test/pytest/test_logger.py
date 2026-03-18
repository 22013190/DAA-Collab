import logging
import os

import pytest

import logger

LOGGER_CLASS = pytest.mark.parametrize(
    "logger_class",
    ["logger_class"],
    indirect=True,
)

ERROR_LOGGER_CLASS = pytest.mark.parametrize(
    "logger_class_error",
    ["logger_class_error"],
    indirect=True,
)


class TestLoggerSuccess:
    def test_path_check(self, request):
        flag = logger._check_path("logs")
        assert flag is True

    def test_log_level_check(self):
        flag = logger._std_log_level("debug")
        assert flag == logging.DEBUG
        flag = logger._std_log_level(123)
        assert flag == logging.DEBUG

    def test_log_path_check(self):
        path = logger._derive_log_path("logs")
        assert os.path.exists(path) is True
        path = logger._derive_log_path("logs/TESTER")
        assert os.path.exists(path) is True
        path = logger._derive_log_path(None)
        assert os.path.exists(path) is True

    @LOGGER_CLASS
    def test_logger_creation(self, logger_class):
        assert logger_class.get_current_logger() is not None
        assert type(logger_class.get_current_logger()) is logging.Logger

    def test_logger_reset_h_log(self, logger_class):
        flag = logger_class.reset_handlers_logger()
        assert flag is True
        log_temp = logger_class.get_current_logger()
        logger_class.logger = None
        flag = logger_class.reset_handlers_logger()
        assert flag is False
        logger_class.logger = log_temp

    def test_logger_add_h(self, logger_class):
        flag = logger_class.add_handler_logger("file_handler")
        assert flag is True

    def test_logger_get_all_h(self, logger_class):
        data = logger_class.get_all_handlers()
        assert data is not None
        assert type(data) is dict

    def test_logger_get_h(self, logger_class):
        data = logger_class.get_handler("file_handler")
        assert data is not None
        assert isinstance(data, logging.Handler) is True

    def test_update_handlers_list(self, logger_class):
        logger_class.update_handlers_list("test_handler", logging.FileHandler)
        assert logger_class.get_handler("test_handler") is not None

    def test_reset_handlers_list(self, logger_class):
        logger_class.reset_handlers_list()
        assert logger_class.get_handler("file_handler") is not None
        assert logger_class.get_handler("console_handler") is not None

    def test_logger_get_all_f(self, logger_class):
        data = logger_class.get_all_formatters()
        assert data is not None
        assert type(data) is dict

    def test_logger_get_f(self, logger_class):
        data = logger_class.get_formatter("default_formatter")
        assert data is not None
        assert isinstance(data, logging.Formatter) is True

    def test_update_formatters_list(self, logger_class):
        logger_class.update_formatters_list("test_formatter", logging.Formatter)
        assert logger_class.get_formatter("test_formatter") is not None

    def test_reset_formatters_list(self, logger_class):
        logger_class.reset_formatters_list()
        assert logger_class.get_formatter("default_formatter") is not None

    def test_setup_logger(self, logger_class):
        logger_class.setup_logger()
        assert logger_class.get_current_logger() is not None
        assert type(logger_class.get_current_logger()) is logging.Logger


class TestLoggerFailures:
    def test_path_check_fail(self):
        flag = logger._check_path(123)
        assert flag is False

    def test_logger_add_h_fail(self, logger_class):
        flag = logger_class.add_handler_logger("rubbish")
        assert flag is False

    def test_logger_get_h_fail(self, logger_class):
        data = logger_class.get_handler("rubbish")
        assert data is not None
        assert isinstance(data, logging.Handler) is True

    def test_get_f_fail(self, logger_class):
        data = logger_class.get_formatter("rubbish")
        assert data is not None
        assert isinstance(data, logging.Formatter) is True
