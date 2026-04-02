"""MCP server for motorsports telemetry coaching."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("inferno-driving-coach")

from . import prompts, tools  # noqa: E402, F401


def main():
    mcp.run()
