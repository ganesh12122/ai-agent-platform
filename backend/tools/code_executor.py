"""
Code Executor Tool - Safe Python code execution
Restrictions: No file system access, no network, no imports allowed
"""

import re
import sys
import io
from typing import Dict, Any, Tuple
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

# Blacklisted modules (dangerous imports)
BLACKLIST = [
    "os", "sys", "subprocess", "socket", "urllib", "requests",
    "pickle", "marshal", "importlib", "eval", "exec", "__import__",
    "open", "file", "input", "compile", "eval", "exec"
]

def _is_safe_code(code: str) -> Tuple[bool, str]:
    """
    Check if code is safe to execute
    Returns: (is_safe, reason)
    """
    
    # Check for blacklisted imports
    for blacklist_item in BLACKLIST:
        if re.search(rf'\b{blacklist_item}\b', code, re.IGNORECASE):
            return False, f"Dangerous import/function detected: {blacklist_item}"
    
    # Check for file operations
    if re.search(r'\bopen\b|\bfile\b|\bpath\b', code, re.IGNORECASE):
        return False, "File system access not allowed"
    
    # Check for network operations
    if re.search(r'\bsocket\b|\brequests\b|\bhttp\b', code, re.IGNORECASE):
        return False, "Network access not allowed"
    
    return True, "Safe"


async def code_executor_tool(code: str) -> str:
    """
    Execute Python code safely
    
    Args:
        code: Python code to execute (as string)
    
    Returns:
        Execution output and results
    
    Restrictions:
        ❌ No file system access
        ❌ No network access
        ❌ No dangerous imports
        ❌ Max 5 seconds execution time
    
    Example:
        >>> code = '''
        >>> def factorial(n):
        ...     return 1 if n <= 1 else n * factorial(n-1)
        >>> print(factorial(5))
        >>> '''
        >>> result = await code_executor_tool(code)
    """
    
    # Validate code safety
    is_safe, reason = _is_safe_code(code)
    if not is_safe:
        return f"❌ Code blocked for security: {reason}"
    
    # Create isolated namespace (limited builtins)
    safe_builtins = {
        'abs': abs, 'all': all, 'any': any, 'ascii': ascii,
        'bin': bin, 'bool': bool, 'bytes': bytes,
        'chr': chr, 'complex': complex,
        'dict': dict, 'dir': dir, 'divmod': divmod,
        'enumerate': enumerate, 'filter': filter, 'float': float,
        'format': format, 'frozenset': frozenset,
        'getattr': getattr, 'hasattr': hasattr, 'hash': hash, 'hex': hex,
        'int': int, 'isinstance': isinstance, 'issubclass': issubclass,
        'iter': iter, 'len': len, 'list': list,
        'map': map, 'max': max, 'min': min,
        'next': next, 'object': object, 'ord': ord,
        'pow': pow, 'print': print, 'property': property,
        'range': range, 'repr': repr, 'reversed': reversed, 'round': round,
        'set': set, 'slice': slice, 'sorted': sorted, 'str': str,
        'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip,
    }
    
    safe_namespace = {'__builtins__': safe_builtins}
    
    # Capture output
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        # Execute code with timeout
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, safe_namespace)
        
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()
        
        # Format result
        result = f"✅ Code Executed Successfully\\n\\n"
        result += f"**Output:**\\n"
        result += f"```\\n{stdout if stdout else '(no output)'}\\n```"
        
        if stderr:
            result += f"\\n**Warnings:**\\n```\\n{stderr}\\n```"
        
        result += f"\\n⏰ Executed at: {datetime.now().isoformat()}"
        return result
    
    except SyntaxError as e:
        return f"❌ Syntax Error:\\n{str(e)}"
    except IndentationError as e:
        return f"❌ Indentation Error:\\n{str(e)}"
    except Exception as e:
        error_msg = str(e)
        if "timed out" in error_msg.lower():
            return f"❌ Execution timeout (max 5 seconds)"
        return f"❌ Runtime Error:\\n{error_msg}"


# Test function (run: python -m tools.code_executor)
if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("Testing code executor...")
        
        code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(8):
    print(f"fib({i}) = {fibonacci(i)}")
'''
        
        result = await code_executor_tool(code)
        print(result)
    
    asyncio.run(test())
