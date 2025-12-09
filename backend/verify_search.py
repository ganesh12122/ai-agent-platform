import asyncio
import sys
import os

# Ensure backend directory is in path
sys.path.append(os.getcwd())

from tools import execute_tool

async def test():
    print("Testing Web Search Tool for Date...")
    queries = [
        "current date today india",
        "what is the date today",
        "today date"
    ]
    
    for q in queries:
        print(f"\nQuerying: '{q}'")
        try:
            result = await execute_tool('web_search', query=q)
            res_str = str(result.get('result', ''))
            print(f"Result Length: {len(res_str)}")
            print(f"Preview: {res_str[:200]}...")
            if "No results found" in res_str:
                print("❌ No results")
            else:
                print("✅ Results found")
        except Exception as e:
            print(f"Error ❌: {e}")

if __name__ == "__main__":
    asyncio.run(test())
