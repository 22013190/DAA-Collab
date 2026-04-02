"""Configurations stored here"""

import importlib
import sys

# Keep imports optional so analysis-only extractions do not hard-require
# unrelated MCP servers (e.g., database).
try:
    from mcp_servers import test_server
except Exception:
    test_server = None

from mcp_servers.analysis import data_processing_server, fastmcp_server

# Keep the database MCP server optional. It may not exist in analysis-only builds.
try:
    db_server = importlib.import_module("mcp_servers.database.db_server")
except ModuleNotFoundError:
    db_server = None
except Exception:
    db_server = None

base_config = {"mcpServers": {}}

# IMPORTANT: `__file__` is evaluated at dict construction time.
# Keep optional servers fully guarded so analysis-only extractions don't crash on import.
if test_server is None:
    test_server_config = {}
else:
    test_server_config = {
        "test": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [test_server.__file__],
            "env": {"DEBUG": "true"},
        }
    }

if db_server is None:
    db_server_config = {}
else:
    db_server_config = {
        "db": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [db_server.__file__],
            "env": {"DEBUG": "true"},
        }
    }

analysis_server_config = {
    "analysis": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [fastmcp_server.__file__],
        "env": {"DEBUG": "true"},
    },
}

data_process_server_config = {
    "data_process": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [data_processing_server.__file__],
        "env": {"DEBUG": "true"},
    },
}


# Help to circumvent circular imports
async def get_base_config() -> dict:
    return base_config


# mcp_config = {
#     "mcpServers": {
#         "test": {
#             "transport": "stdio",
#             "command": "python",
#             "args": [test_server.__file__],
#             "env": {"DEBUG": "true"},
#         },
#         "database": {
#             "transport": "stdio",
#             "command": "python",
#             "args": [db_server.__file__],
#             "env": {"DEBUG": "true"},
#         },
#     }
# }
