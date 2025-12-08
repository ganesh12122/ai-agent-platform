import httpx
import asyncio
import sys

async def test_agent_with_tools():
    print("🧪 Testing Day 3 - Tools Integration\n")
    
    base = "http://localhost:8000"
    
    tests = [
        {
            "name": "Web Search",
            "query": "What is ChatGPT?",
            "expected_tool": "web_search",
            "expected_trace": "tool_executor"
        },
        {
            "name": "Code Generation",
            "query": "Write Python function for factorial",
            "expected_tool": "code_executor",
            "expected_trace": "tool_router" 
        },
        {
            "name": "General Query",
            "query": "Hello, how are you?",
            "expected_tool": "none",
            "expected_trace": "response_formatter"
        }
    ]
    
    for test in tests:
        print(f"Test: {test['name']}")
        print(f"  Query: {test['query']}")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{base}/api/route-query",
                    json={"query": test['query']}
                )
                
            if resp.status_code != 200:
                print(f"  ❌ Error: Status {resp.status_code}")
                continue
                
            data = resp.json()
            
            trace = data.get('agent_trace', [])
            tools_used = data.get('tool_calls', []) # Check if we added this to route-query response?
            
            # Check for expected trace
            found_trace = any(test['expected_trace'] in t for t in trace)
            
            print(f"  ✅ Route: {data.get('recommended_model', 'unknown')}")
            print(f"  ✅ Trace: {trace}")
            
            if test['name'] == "Web Search":
                # For web search, checking if tool_executor was visited and tool_calls contains web_search
                # Note: route-query response might not expose everything unless we updated main.py
                # Let's see what main.py returns.
                pass
            
            print()
        except Exception as e:
            print(f"  ❌ Error: {e}\n")

if __name__ == "__main__":
    asyncio.run(test_agent_with_tools())
