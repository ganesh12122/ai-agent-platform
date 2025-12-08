"""
Web Search Tool - Uses DuckDuckGo API
Safe, free, no API key required
"""

import httpx
import json
from typing import List, Dict, Any
from datetime import datetime

# DuckDuckGo Instant API (Free, no auth needed)
DDGO_API_URL = "https://api.duckduckgo.com"

async def web_search_tool(query: str, num_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo API
    
    Args:
        query: Search query string
        num_results: Number of results to return (max 10)
    
    Returns:
        Formatted search results as string
    
    Example:
        >>> result = await web_search_tool("latest AI news")
        >>> print(result)
        "1. Title - snippet - url"
    """
    
    if not query or len(query.strip()) < 3:
        return "❌ Search query too short (min 3 characters)"
    
    try:
        # Call DuckDuckGo API
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(DDGO_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
        
        # Extract results
        results = []
        
        # Get instant answer if available
        if data.get("AbstractText"):
            results.append({
                "title": "Abstract",
                "snippet": data.get("AbstractText", ""),
                "url": data.get("AbstractURL", "")
            })
        
        # Get related topics/results
        related_topics = data.get("RelatedTopics", [])
        for topic in related_topics[:num_results]:
            if isinstance(topic, dict):
                results.append({
                    "title": topic.get("FirstURL", "").split("/")[-1] or "Result",
                    "snippet": topic.get("Text", ""),
                    "url": topic.get("FirstURL", "")
                })
        
        # Format results
        if not results:
            return f"No results found for: {query}"
        
        formatted = f"🔍 Search Results for '{query}'\\n\\n"
        for i, result in enumerate(results[:num_results], 1):
            formatted += f"{i}. **{result['title']}**\\n"
            formatted += f"   {result['snippet'][:200]}...\\n"
            formatted += f"   🔗 {result['url']}\\n\\n"
        
        formatted += f"\\n⏰ Searched at: {datetime.now().isoformat()}"
        return formatted
    
    except httpx.TimeoutException:
        return "❌ Search timeout - DuckDuckGo took too long"
    except Exception as e:
        return f"❌ Search error: {str(e)}"


# Test function (run: python -m tools.web_search)
if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("Testing web search...")
        result = await web_search_tool("Python 3.12 release date")
        print(result)
    
    asyncio.run(test())
