import asyncio
import sys
import os

# Ensure backend directory is in path
sys.path.append(os.getcwd())

from tools import execute_tool

async def test():
    print("Testing Web Search Tool...")
    try:
        result = await execute_tool('web_search', query='latest AI news')
        print(f"Result Type: {type(result)}")
        print(f"Result Keys: {result.keys()}")
        print(f"Content Length: {len(str(result.get('result', '')))}")
        print("Success ✅")
    except Exception as e:
        print(f"Error ❌: {e}")

if __name__ == "__main__":
    asyncio.run(test())
