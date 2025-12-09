"""
Web Search Tool - Uses DuckDuckGo API
Safe, free, no API key required
"""

import asyncio
from duckduckgo_search import DDGS
from datetime import datetime

async def web_search_tool(query: str, num_results: int = 5) -> str:
    """
    Search the web using the robust duckduckgo-search package.
    """
    if not query or len(query.strip()) < 3:
        return "❌ Search query too short (min 3 characters)"
    
    try:
        # Use DDGS for search
        # Run in executor to avoid blocking async loop since DDGS is sync/blocking by default in older versions
        # or use the async context manager if available (latest version supports async)
        
        # Checking latest docs: DDGS().text() is synchronous. We should run it in a thread.
        # However, for simplicity let's try direct call first, or better wrap in to_thread.
        
        results = []
        
        # Synchronous call wrapping
        def run_search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=num_results))
                
        # Run in thread pool
        raw_results = await asyncio.to_thread(run_search)
        
        if not raw_results:
            return f"No results found for: {query}"
            
        formatted = f"🔍 Search Results for '{query}'\\n\\n"
        for i, res in enumerate(raw_results, 1):
            title = res.get('title', 'No Title')
            snippet = res.get('body', '') or res.get('snippet', '')
            url = res.get('href', '') or res.get('link', '')
            
            formatted += f"{i}. **{title}**\\n"
            formatted += f"   {snippet[:200]}...\\n"
            formatted += f"   🔗 {url}\\n\\n"
            
        formatted += f"\\n⏰ Searched at: {datetime.now().isoformat()}"
        return formatted
        
    except Exception as e:
        return f"❌ Search error: {str(e)}"

if __name__ == "__main__":
    async def test():
        print("Testing web search...")
        # result = await web_search_tool("Python 3.12 release date") # Commented out to avoid auto-run
        # print(result)
        pass
    # asyncio.run(test())
