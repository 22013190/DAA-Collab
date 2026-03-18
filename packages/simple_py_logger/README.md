# Simple Python Logger

This is a simple python logger class created using stdlib `logging` with `Pydantic` for data validation and `Rich` for colourful text formatting.
No other dependencies used in this module.

# Usage

```py
import <path>.<to>.logger

parent_logger = Logger("test")
logger = parent_logger.get_current_logger()

logger.debug("Here are some errors")
```

Output:

```bash
[06/12/25 15:29:06] DEBUG    test_2025-06-12 15:29:06,677 LEVEL: DEBUG -                                      logger.py:411
                             /<path_to>/src/logger.py + MESSAGE: bug is found
```

The logger class contains 2 dictionary object **formatters** and **handlers** which allows storage of their custom object types.

A formatter should be created first if needed. Subsequently, attached the formatter into the handler to have your custom output format. Use [helpers](##handler_helpers) for some handler creation functions or you can create your own custom handlers.

Any logger that are instantiated would have [defaults](#Defaults) and custom formatter/handler added into the list will not be reflected in the current instance.

Use `setup_logger()` or manually add them via `add_handler_logger()` into the existing logger instance.

# Defaults

After initialisation of class, there are some default formatter and handlers in place to ease setup.

Formatters would have a key-Value pair of **DEFAULT_FORMATTER** and `{name}_{asctime} LEVEL: {levelname} - {pathname} + MESSAGE: {message}`

Handlers would make use of **DEFAULT_FORMATTER** and create 2 Key-Value handlers for _console_ and _file_ logging.

# APIs

## Logger Class Methods

```py
    def reset_handlers_logger(self) -> bool

    def remove_handler_logger(self, handler_key: str) -> bool

    def add_handler_logger(self, handler_key: str) -> bool

    def get_all_handlers(self) -> dict[str, Type[logging.Handler]] | None

    def get_handler(self, handler_key: str) -> Type[logging.Handler]

    def update_handlers_list(
        self, handler_key: str, handler: Type[logging.Handler]
    ) -> None

    def reset_handlers_list(self) -> None

    def get_all_formatters(self) -> dict[str, logging.Formatter] | None

    def get_formatter(self, formatter_key: str) -> logging.Formatter

    def update_formatters_list(
        self, formatter_key: str, formatter: Type[logging.Formatter]
    ) -> None

    def reset_formatters_list(self) -> None

    def get_current_logger(self) -> logging.Logger

    def setup_logger(
        self,
        module_name: str | None = None,
        log_level: str | None = None,
    ) -> None

```

## handler_helpers

```py
def create_console_handler(
    log_level_int: int,
    formatter: logging.Formatter,
    stream=sys.stdout,
) -> logging.StreamHandler

def create_rich_console_handler(
    log_level_int: int,
    formatter: logging.Formatter,
    markup: bool = True,
    rich_tracebacks: bool = True,
) -> RichHandler

def create_file_handler(
    log_level_int: int,
    formatter: logging.Formatter,
    log_path: str,
    mode: str = "a",
    encoding: str | None = None,
    delay: bool = False,
    errors: str | None = None,
) -> logging.FileHandler

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
) -> log_handlers.RotatingFileHandler

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
) -> log_handlers.TimedRotatingFileHandler

def create_syslog_handler(
    log_level_int: int,
    formatter: logging.Formatter,
    address: Tuple[str, int] = ("localhost", log_handlers.SYSLOG_UDP_PORT),
    facility: int = log_handlers.SysLogHandler.LOG_USER,
    socktype: socket.SocketKind = socket.SOCK_DGRAM,
) -> log_handlers.SysLogHandler

def create_http_handler(
    log_level_int: int,
    formatter: logging.Formatter,
    host: str,
    url: str,
    method: Literal["GET", "POST"] = "GET",
    secure: bool = False,
    credentials: Tuple[str, str] | None = None,
    context: ssl.SSLContext | None = None,
) -> log_handlers.HTTPHandler

```

# Run Tests

From root dir of the project run `uv run pytest`.
