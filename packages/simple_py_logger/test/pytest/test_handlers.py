import logging
import logging.handlers
import sys

import pytest
from rich.logging import RichHandler

import handler_helpers as handlers

BASE_PARAMS = pytest.mark.parametrize(
    "handler_params", ["handler_params"], indirect=True
)


@BASE_PARAMS
class TestHandlerSuccess:
    def test_console_creation(self, handler_params, request, stream=sys.stderr):
        console_h = handlers.create_console_handler(
            handler_params["log_level_int"],
            handler_params["formatter"],
            stream=stream,
        )
        assert type(console_h) is logging.StreamHandler

    @pytest.mark.parametrize("rich_params", ["rich_params"], indirect=True)
    def test_rich_console_creation(self, handler_params, rich_params, request):
        rich_h = handlers.create_rich_console_handler(
            handler_params["log_level_int"],
            handler_params["formatter"],
            rich_params["markup"],
            rich_params["rich_tracebacks"],
        )
        assert type(rich_h) is RichHandler

    @pytest.mark.parametrize("file_params", ["file_params"], indirect=True)
    def test_rich_creation(self, handler_params, file_params, request):
        file_h = handlers.create_file_handler(
            handler_params["log_level_int"],
            handler_params["formatter"],
            file_params["log_path"] + "/" + request.cls.__name__,
            file_params["mode"],
            file_params["encoding"],
            file_params["delay"],
            file_params["errors"],
        )
        assert type(file_h) is logging.FileHandler

    @pytest.mark.parametrize(
        "rotate_file_params", ["rotate_file_params"], indirect=True
    )
    def test_rotate_file_creation(self, handler_params, rotate_file_params, request):
        rotate_file_h = handlers.create_rotate_file_handler(
            handler_params["log_level_int"],
            handler_params["formatter"],
            rotate_file_params["log_path"] + "/" + request.cls.__name__,
            rotate_file_params["mode"],
            rotate_file_params["maxBytes"],
            rotate_file_params["backupCount"],
            rotate_file_params["encoding"],
            rotate_file_params["delay"],
            rotate_file_params["errors"],
        )
        assert type(rotate_file_h) is logging.handlers.RotatingFileHandler

    @pytest.mark.parametrize(
        "rotate_time_params", ["rotate_time_params"], indirect=True
    )
    def test_rotate_time_creation(self, handler_params, rotate_time_params, request):
        rotate_time_h = handlers.create_rotate_time_handler(
            handler_params["log_level_int"],
            handler_params["formatter"],
            rotate_time_params["log_path"] + "/" + request.cls.__name__,
            rotate_time_params["when"],
            rotate_time_params["interval"],
            rotate_time_params["backupCount"],
            rotate_time_params["encoding"],
            rotate_time_params["delay"],
            rotate_time_params["utc"],
            rotate_time_params["atTime"],
            rotate_time_params["errors"],
        )
        assert type(rotate_time_h) is logging.handlers.TimedRotatingFileHandler

    @pytest.mark.parametrize("syslog_params", ["syslog_params"], indirect=True)
    def test_syslog_creation(self, handler_params, syslog_params, request):
        syslog_h = handlers.create_syslog_handler(
            handler_params["log_level_int"],
            handler_params["formatter"],
            syslog_params["address"],
            syslog_params["facility"],
            syslog_params["socktype"],
        )
        assert type(syslog_h) is logging.handlers.SysLogHandler

    @pytest.mark.parametrize("http_params", ["http_params"], indirect=True)
    def test_http_creation(self, handler_params, http_params, request):
        http_h = handlers.create_http_handler(
            handler_params["log_level_int"],
            handler_params["formatter"],
            http_params["host"],
            http_params["url"],
            http_params["method"],
            http_params["secure"],
            http_params["credentials"],
            http_params["context"],
        )
        assert type(http_h) is logging.handlers.HTTPHandler


@BASE_PARAMS
class TestHandlerFailures:
    def test_console_creation_fail(self, handler_params, request, stream=""):
        with pytest.raises(Exception) as exc_info:
            fail_h = handlers.create_console_handler(
                handler_params["log_level_int"],
                handler_params["formatter"],
                stream=stream,
            )

    @pytest.mark.parametrize("rich_params", ["rich_params"], indirect=True)
    def test_rich_console_fail(self, handler_params, rich_params, request):
        with pytest.raises(Exception) as exc_info:
            fail_h = handlers.create_rich_console_handler(
                handler_params["log_level_int"],
                handler_params["formatter"],
                rich_params["markup"],
                rich_params["rich_tracebacks"],
            )

    @pytest.mark.parametrize("file_params", ["file_params"], indirect=True)
    def test_file_creation_fail(self, handler_params, file_params, request):
        with pytest.raises(Exception) as exc_info:
            fail_h = handlers.create_file_handler(
                handler_params["log_level_int"],
                handler_params["formatter"],
                file_params["log_path"] + "/" + request.cls.__name__,
                file_params["mode"],
                file_params["encoding"],
                file_params["delay"],
                file_params["errors"],
            )

    @pytest.mark.parametrize(
        "rotate_file_params", ["rotate_file_params"], indirect=True
    )
    def test_rotate_file_creation_fail(
        self, handler_params, rotate_file_params, request
    ):
        with pytest.raises(Exception) as exc_info:
            fail_h = handlers.create_rotate_file_handler(
                handler_params["log_level_int"],
                handler_params["formatter"],
                rotate_file_params["log_path"] + "/" + request.cls.__name__,
                rotate_file_params["mode"],
                rotate_file_params["maxBytes"],
                rotate_file_params["backupCount"],
                rotate_file_params["encoding"],
                rotate_file_params["delay"],
                rotate_file_params["errors"],
            )

    @pytest.mark.parametrize(
        "rotate_time_params", ["rotate_time_params"], indirect=True
    )
    def test_rotate_time_creation_fail(
        self, handler_params, rotate_time_params, request
    ):
        with pytest.raises(Exception) as exc_info:
            fail_h = handlers.create_rotate_time_handler(
                handler_params["log_level_int"],
                handler_params["formatter"],
                rotate_time_params["log_path"] + "/" + request.cls.__name__,
                rotate_time_params["when"],
                rotate_time_params["interval"],
                rotate_time_params["backupCount"],
                rotate_time_params["encoding"],
                rotate_time_params["delay"],
                rotate_time_params["utc"],
                rotate_time_params["atTime"],
                rotate_time_params["errors"],
            )

    @pytest.mark.parametrize("syslog_params", ["syslog_params"], indirect=True)
    def test_syslog_creation_fail(self, handler_params, syslog_params, request):
        with pytest.raises(Exception) as exc_info:
            fail_h = handlers.create_syslog_handler(
                handler_params["log_level_int"],
                handler_params["formatter"],
                syslog_params["address"],
                syslog_params["facility"],
                syslog_params["socktype"],
            )

    @pytest.mark.parametrize("http_params", ["http_params"], indirect=True)
    def test_http_creation_fail(self, handler_params, http_params, request):
        with pytest.raises(Exception) as exc_info:
            fail_h = handlers.create_http_handler(
                handler_params["log_level_int"],
                handler_params["formatter"],
                http_params["host"],
                http_params["url"],
                http_params["method"],
                http_params["secure"],
                http_params["credentials"],
                http_params["context"],
            )
