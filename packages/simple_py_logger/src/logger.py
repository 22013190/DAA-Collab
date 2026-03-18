"""
This module uses Python's standard logging library while using Pydantic for data
validation and Rich for colourful formatting.

The goal is to be a simple extensible logger that can be attached to any modules
that require logging of data in the process.

Currently, no support for filter object instances due to low/no usages at all.

Advanced usages should refer to: https://docs.python.org/3/howto/logging-cookbook.html#
"""

import logging
import os
from typing import Annotated, ClassVar, Type

import rich.repr
from pydantic import BaseModel, ConfigDict, Field

from handler_helpers import (
    create_console_handler,
    create_file_handler,
    create_rich_console_handler,
)


def _check_path(user_path: str) -> bool:
    """
    Check if path exists

    Args:
        user_path: Path value from user input

    Returns:
        bool: Boolean value if path exists
    """
    if not os.path.exists(user_path):
        return False
    return True


def _std_log_level(log_level_str: str | None) -> int:
    """
    Gets log level integer value from string input

    Args:
        log_level_str: String value of log level names.

    Returns:
        int: Log level number for logger instance. Defaults to INFO level.
    """
    if log_level_str is None:
        log_level_str = "DEBUG"
    if type(log_level_str) is not str:
        log_level_str = "DEBUG"
    match log_level_str.upper():
        case "DEBUG":
            return logging.DEBUG
        case "INFO":
            return logging.INFO
        case "WARNING":
            return logging.WARNING
        case "ERROR":
            return logging.ERROR
        case "CRITICAL":
            return logging.CRITICAL
        case _:
            return logging.INFO


def _derive_log_path(log_path: str | None) -> str:
    """
    Create log dir path based on user input. If path does not exists,
    create a log dir path based on current execution path.

    Args:
        log_path:   User input of desired log location relative to their current
                    execution path.

    Returns:
        str: The log path string for storage of logs.
    """
    # Get current execution path
    cwd_path = os.getcwd()

    if log_path is None or not _check_path(cwd_path + "/" + log_path):
        # Create logfile and dir if needed
        log_dir = cwd_path + "/logs"

        try:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Error resolving log path: {e}")
        return log_dir
    else:
        return cwd_path + "/" + log_path


class Logger(BaseModel):
    """
    Parent logger class deriving from root logger of python's stdlib

    Attributes:
        module_name  : String name to represent instantiated class.
        log_path     : Custom file path preferrably to `/logs` directory.
        log_level    : String value based on Python's std log levels.
        log_level_int: Integer value derived from `log_level`.
        logger       : Logger instance using Python's std lib.
        formatters   : A dict of formatter instances and their string keys.
        handlers     : A dict of handler instances and their string keys.
    """

    model_config = ConfigDict(
        validate_default=True,
        arbitrary_types_allowed=True,
    )
    DEFAULT_FORMATTER: ClassVar[logging.Formatter] = logging.Formatter(
        "{name}_{asctime} LEVEL: {levelname} - {pathname} + MESSAGE: {message}",
        style="{",
    )

    module_name: Annotated[str, Field(default="default_logger", init_var=True)]
    log_path: Annotated[str | None, Field(default=None, strict=True)]
    log_level: Annotated[str, Field(default="DEBUG")]
    log_level_int: Annotated[int, Field(default=logging.DEBUG)]
    logger: Annotated[logging.Logger | None, Field(default=None)]
    formatters: Annotated[
        dict[str, logging.Formatter] | None, Field(default={}, strict=False)
    ] = {}
    handlers: Annotated[
        dict[str, Type[logging.Handler]] | None, Field(default={}, strict=False)
    ] = {}

    def __init__(
        self,
        module_name: str = "default_logger",
        log_path: str | None = None,
        log_level: str | None = "DEBUG",
    ) -> None:
        """
        Initialises the class instance.

        Args:
            module_name: Name to be attached to the logger instance.
            log_path: Directory path from where the module is being executed from.
                The directory should be created beforehand otherwise, it will use your current
                execution path location as relative location.
                Defaults to None. Format: "<path>/<to>/<dst>".
            log_level: string value of standard log levels. Defaults to DEBUG level.
            formatter: Custom formatter instance otherwise uses DEFAULT_FORMATTER.
        """
        # Initialise values
        super().__init__(
            module_name=module_name,
            log_path=_derive_log_path(log_path),
            log_level=str(log_level).upper(),
            log_level_int=_std_log_level(log_level),
            handlers={},
            formatters={"default_formatter": self.DEFAULT_FORMATTER},
        )

        # Initialise default handlers and base logger
        # self.handlers["console_handler"] = create_console_handler(
        #     log_level_int=self.log_level_int,
        #     formatter=self.formatters["default_formatter"],
        # )
        self.handlers["rich_console_handler"] = create_rich_console_handler(
            log_level_int=self.log_level_int,
            formatter=self.formatters["default_formatter"],
        )
        self.handlers["file_handler"] = create_file_handler(
            log_level_int=self.log_level_int,
            formatter=self.formatters["default_formatter"],
            log_path=self.log_path + "/" + self.module_name + ".log",
        )
        self.setup_logger(self.module_name, self.log_level)

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.module_name
        yield "Logging to", self.log_path
        yield "Logging level", self.log_level
        yield "Formatters", self.formatters
        yield "Handlers", self.handlers
        yield "Logging Instance", self.logger

    def reset_handlers_logger(self) -> bool:
        """
        Remove all handlers within the logger instance.

        Returns:
            bool: True for removing handlers and False when list is currently
                empty.
        """
        if self.logger:
            if not self.logger.handlers:
                return True
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
            return True
        return False

    def remove_handler_logger(self, handler_key: str) -> bool:
        """
        Removes handler attached to logger.

        This requires a handler object that exists within `handlers` attribute.

        Args:
            handler_key: Key string to check the handler dictionary.

        Returns:
            bool: True if object is found and removed otherwise False.
        """
        if handler_key in self.handlers:
            self.logger.removeHandler(self.handlers[handler_key])
            return True
        return False

    def add_handler_logger(self, handler_key: str) -> bool:
        """
        This only add handler into existing list of handlers. Instantiated class
        would not have the new handlers added, requiring separate manual configuration.

        This adds the handler value from the dictionary into the existing

        Args:
            handler_key: Key string to check the handler dictionary.
        """
        if handler_key in self.handlers:
            self.logger.addHandler(self.handlers[handler_key])
            return True
        return False

    def get_all_handlers(self) -> dict[str, Type[logging.Handler]] | None:
        """
        Returns current dictionary of handlers

        Returns:
            dict[str, logging.Handler]: An object of existing handlers
        """
        return self.handlers

    def get_handler(self, handler_key: str) -> Type[logging.Handler]:
        """
        Get specific handler from dictionary object.

        Returns `rich_console_handler`(priority), `file_handler` or `console_handler`
        by default if key does not exists.

        Args:
            handler_key: String value to search in dictionary.

        Returns:
            logging.Handler: The handler object found within the object.
        """
        try:
            h = self.handlers[handler_key]
        except KeyError as k:
            # Return as default
            print(
                f"""Warning: obtaining handler that does not exists: {k}
    Returning default console handler instead."""
            )
            if "console_handler" in self.handlers:
                h = self.handlers["console_handler"]
            if "file_handler" in self.handlers:
                h = self.handlers["file_handler"]
            if "rich_console_handler" in self.handlers:
                h = self.handlers["rich_console_handler"]
        except Exception as e:
            print(f"Error obtaining handler: {e}")
        return h

    def update_handlers_list(
        self, handler_key: str, handler: Type[logging.Handler]
    ) -> None:
        """
        This only add handler into existing list of handlers. If there is an existing
        key-value pair, the new handler will override the existing handler value.

        Instantiated class would not have the new handlers added, use
        `add_handler_logger` to include into the instance.

        Args:
            handler_key: A string key to pair with the handler value
            handler    : Base handler/subclass handlers. More
                details in: https://docs.python.org/3/howto/logging.html#useful-handlers
        """
        self.handlers[handler_key] = handler

    def reset_handlers_list(self) -> None:
        """
        Empty the list of handlers within the class instance.
        This does not impact the existing handlers that have been instantiated.
        After reset, default console and file handlers will be created in the dictionary.
        """
        self.handlers = {}
        self.handlers["console_handler"] = create_console_handler(
            log_level_int=self.log_level_int,
            formatter=self.formatters["default_formatter"],
        )
        self.handlers["file_handler"] = create_file_handler(
            log_level_int=self.log_level_int,
            formatter=self.formatters["default_formatter"],
            log_path=self.log_path + "/" + self.module_name,
        )

    def get_all_formatters(self) -> dict[str, logging.Formatter] | None:
        """
        Returns current dictionary of formatters

        Returns:
            dict[str, logging.Formatter]: An object of existing formatters
        """
        return self.formatters

    def get_formatter(self, formatter_key: str | None) -> logging.Formatter:
        """
        Get specific formatter from dictionary object.

        Returns `default_formatter` by default if key does not exists.

        Args:
            formatter_key: String value to search in dictionary.

        Returns:
            logging.Formatter: The formatter object found within the object.
        """
        try:
            f = self.formatters[formatter_key]
        except KeyError as k:
            # Return as default
            print(
                f"""Warning: obtaining formatter that does not exists: {k}
    Returning default formatter instead."""
            )
            f = self.formatters["default_formatter"]
        except Exception as e:
            print(f"Error obtaining formatter: {e}")
        return f

    def update_formatters_list(
        self, formatter_key: str, formatter: Type[logging.Formatter]
    ) -> None:
        """
        This only add formatter into existing list of formatters. If there is an existing
        key-value pair, the new handler will override the existing handler value.

        Usage of formatter should be handled by developer as it is tied with handler
        creation.

        Args:
            formatter_key:  A string key to pair with the formatter value
            formatter    :  Base formmater value. More details
                            in: https://docs.python.org/3/howto/logging.html#formatters
        """
        self.formatters[formatter_key] = formatter

    def reset_formatters_list(self) -> None:
        """
        Empty the list of handlers within the class instance and add default formatter.
        This does not impact the existing handlers that have been instantiated.
        """
        self.formatters = {}
        self.formatters["default_formatter"] = self.DEFAULT_FORMATTER

    def get_current_logger(self) -> logging.Logger:
        return self.logger

    def setup_logger(
        self,
        module_name: str | None = None,
        log_level: str | None = None,
    ) -> None:
        """
        Sets up logger for the class. Only one instance can be initialised.

        Calling this function again will initialise a new logger creation and
        override the existing instance. This adds all handler objects into
        the instance.

        Args:
            module_name:    Name to be attached to logger instance. File creation will
                            depend on this name as well.
            log_level  :    Log level names based on standard library values.
            formatter  :    Formatter instance to attach to handlers.

        Returns:
            logging.Logger: The logger instance attached to the class.
        """
        # Create logger
        logger = logging.getLogger(module_name or self.module_name)
        logger.setLevel(_std_log_level(log_level) or self.log_level)

        # Clear handlers in logger
        self.reset_handlers_logger()

        # Do not propagate logs into root logger
        logger.propagate = False

        # Add handlers
        for key in self.handlers:
            logger.addHandler(self.handlers[key])

        self.logger = logger


# # Debug purposes
# if __name__ == "__main__":
#     log = Logger("test", None, "debug")
#     logger = log.get_current_logger()
#     logger.debug("bug is found")
#     logger.error("has error")
