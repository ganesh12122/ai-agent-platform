"""
Tool Registry for the AI Agent
Exposes all available tools through a registry
"""

from .web_search import web_search_tool
from .code_executor import code_executor_tool

# Tool Registry - Maps tool names to functions
TOOLS_REGISTRY = {
    "web_search": {
        "function": web_search_tool,
        "description": "Search the web for information using DuckDuckGo",
        "input_format": "query (str)",
        "when_to_use": "User asks for current events, recent news, or external information"
    },
    "code_executor": {
        "function": code_executor_tool,
        "description": "Execute Python code safely and return results",
        "input_format": "code (str)",
        "when_to_use": "User asks to write, test, or run Python code"
    }
}

async def execute_tool(tool_name: str, **kwargs) -> dict:
    """
    Execute a tool by name
    
    Args:
        tool_name: Name of tool from TOOLS_REGISTRY
        **kwargs: Arguments to pass to tool
    
    Returns:
        {"status": "success|error", "result": "...", "tool": "..."}
    """
    if tool_name not in TOOLS_REGISTRY:
        return {
            "status": "error",
            "result": f"Tool '{tool_name}' not found",
            "tool": tool_name
        }
    
    try:
        tool_func = TOOLS_REGISTRY[tool_name]["function"]
        result = await tool_func(**kwargs)
        return {
            "status": "success",
            "result": result,
            "tool": tool_name
        }
    except Exception as e:
        return {
            "status": "error",
            "result": str(e),
            "tool": tool_name
        }

# Export for main.py
__all__ = ["TOOLS_REGISTRY", "execute_tool"]
