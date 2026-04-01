from fastmcp import FastMCP

server = FastMCP("Test")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


if __name__ == "__main__":
    server.run(transport="stdio")
