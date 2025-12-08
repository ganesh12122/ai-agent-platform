import os
import operator
from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph import StateGraph, END
import httpx
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

from tools import execute_tool

# ============================================================================
# 1. DEFINE AGENT STATE
# ============================================================================
# The state is passed between nodes in the graph. 
# It serves as the shared memory for the agent.

class AgentState(TypedDict):
    """
    Represents the state of the multi-model AI agent.
    """
    messages: List[Dict[str, str]]        # History of messages: [{"role": "user", "content": "..."}]
    model: str                            # The model selected for execution (e.g., "mistral", "deepseek-coder")
    intent: str                           # Detected intent (e.g., "code", "general")
    tools_used: List[str]                 # Track tools/nodes executed
    tool_calls: List[str]                 # Track tool execution
    tool_results: Dict[str, Any]          # Results from tools
    conversation_context: Dict[str, Any]  # Arbitrary context data
    request_id: str                       # Unique ID for tracing
    final_response: str                   # The generated response to return

# ============================================================================
# 2. DEFINE NODES
# ============================================================================
# Nodes are functions that process the state and return updates.

async def query_analyzer_node(state: AgentState) -> Dict[str, Any]:
    """
    Node A: Query Analyzer
    Analyzes the latest user message to determine intent (Code vs General).
    """
    print("--- 🔍 Analyzing Query ---")
    
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "general"}
    
    last_message = messages[-1]["content"].lower()
    
    # Simple keyword-based intent detection
    # In a real app, you might use a lightweight LLM or classifier here
    code_keywords = ["code", "python", "function", "bug", "error", "api", "json", "debug"]
    
    if any(keyword in last_message for keyword in code_keywords):
        intent = "code"
    else:
        intent = "general"
        
    print(f"Detected Intent: {intent}")
    
    # Return updates to the state
    return {
        "intent": intent, 
        "tools_used": ["query_analyzer"]
    }

async def model_selector_node(state: AgentState) -> Dict[str, Any]:
    """
    Node B: Model Selector
    Selects the best model based on the analyzed intent.
    """
    print("--- 🤖 Selecting Model ---")
    
    intent = state.get("intent", "general")
    
    if intent == "code":
        model = "deepseek-coder"
    else:
        model = "mistral"
        
    print(f"Selected Model: {model}")
    
    # Update tools_used and set the model in state
    current_tools = state.get("tools_used", [])
    return {
        "model": model,
        "tools_used": current_tools + ["model_selector"]
    }

async def model_executor_node(state: AgentState) -> Dict[str, Any]:
    """
    Node C: Model Executor
    Calls the Ollama API with the selected model to generate a response.
    """
    print("--- ⚡ Executing Model ---")
    
    model = state.get("model", "mistral")
    messages = state.get("messages", [])
    tool_results = state.get("tool_results", {})
    
    # Build prompt with tool results
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    
    # Add tool results as context
    if tool_results:
        prompt += "\n\n[Tool Results Available]:\n"
        for tool_name, result in tool_results.items():
            prompt += f"\n{tool_name}:\n{result}\n"
        prompt += "\nUse these results to answer the user's question.\n"
    
    payload = {
        "model": model, 
        "prompt": prompt,
        "stream": False
    }
    
    response_text = ""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # We strip the tag if it exists for the API call if needed, 
            # but usually Ollama expects 'name:tag'.
            # If deepseek isn't pulled, this might fail, so we fallback to mistral in a real app.
            
            resp = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                response_text = data.get("response", "")
            else:
                response_text = f"Error: Ollama API returned {resp.status_code}"
                
    except Exception as e:
        response_text = f"Error executing model: {str(e)}"
        
    print(f"Generated Response Length: {len(response_text)}")
    
    current_tools = state.get("tools_used", [])
    return {
        "final_response": response_text,
        "tools_used": current_tools + ["model_executor"]
    }

async def response_formatter_node(state: AgentState) -> Dict[str, Any]:
    """
    Node D: Response Formatter
    Formats the final response, potentially adding metadata or structuring JSON.
    """
    print("--- 📝 Formatting Response ---")
    
    raw_response = state.get("final_response", "")
    metadata = {
        "model_used": state.get("model"),
        "intent_detected": state.get("intent"),
        "processed_by": state.get("tools_used"),
        "tool_calls": state.get("tool_calls", [])
    }
    
    # In this simple example, we just append metadata to the context
    # or leave the text as is. Let's just update the context.
    
    current_tools = state.get("tools_used", [])
    return {
        "conversation_context": metadata,
        "tools_used": current_tools + ["response_formatter"]
    }

async def tool_router_node(state: AgentState) -> Dict[str, Any]:
    """
    Node E: Tool Router
    
    Decides if we need to use a tool based on query intent
    Routes to appropriate tool or skips to LLM
    """
    
    print("--- 🔧 Tool Router ---")
    
    intent = state.get("intent", "general")
    messages = state.get("messages", [])
    last_message = messages[-1]["content"].lower() if messages else ""
    
    tool_calls = []
    
    # Decide which tools to use
    needs_web_search = any(keyword in last_message for keyword in [
        "search", "news", "latest", "current", "today", "recent",
        "what is", "who is", "weather", "stock", "price"
    ])
    
    needs_code_exec = intent == "code"
    
    # Execute tools
    if needs_web_search:
        print(f"  → Using web search for: {last_message[:50]}...")
        tool_calls.append("web_search")
        # Will execute in tool_executor_node
    
    if needs_code_exec:
        print(f"  → Code execution may be needed")
        tool_calls.append("code_executor")
    
    current_tools = state.get("tools_used", [])
    
    return {
        "tool_calls": tool_calls,
        "tools_used": current_tools + ["tool_router"]
    }


async def tool_executor_node(state: AgentState) -> Dict[str, Any]:
    """
    Node F: Tool Executor
    
    Actually executes the selected tools
    Stores results in state for LLM to use
    """
    
    print("--- 🛠️ Tool Executor ---")
    
    tool_calls = state.get("tool_calls", [])
    tool_results = {}
    
    messages = state.get("messages", [])
    query = messages[-1]["content"] if messages else ""
    
    # Execute web search if needed
    if "web_search" in tool_calls:
        print(f"  → Executing web search...")
        try:
            result = await execute_tool("web_search", query=query)
            tool_results["web_search"] = result.get("result", "")
        except Exception as e:
            tool_results["web_search"] = f"Web search failed: {str(e)}"
    
    # Execute code if needed
    if "code_executor" in tool_calls:
        print(f"  → Code needs to be generated by LLM first")
        # We'll handle this after LLM generation
    
    current_tools = state.get("tools_used", [])
    
    return {
        "tool_results": tool_results,
        "tools_used": current_tools + ["tool_executor"]
    }

# ============================================================================
# 3. BUILD THE GRAPH
# ============================================================================

# Initialize the StateGraph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("query_analyzer", query_analyzer_node)
workflow.add_node("model_selector", model_selector_node)
workflow.add_node("tool_router", tool_router_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("model_executor", model_executor_node)
workflow.add_node("response_formatter", response_formatter_node)

# Add edges to define the flow
# Entry Point -> Analyzer
workflow.set_entry_point("query_analyzer")

# Analyzer -> Selector
workflow.add_edge("query_analyzer", "model_selector")

# Selector -> Tool Router
workflow.add_edge("model_selector", "tool_router")

# Tool Router -> Tool Executor
workflow.add_edge("tool_router", "tool_executor")

# Tool Executor -> Model Executor (LLM)
workflow.add_edge("tool_executor", "model_executor")

# Model Executor -> Formatter
workflow.add_edge("model_executor", "response_formatter")

# Formatter -> End
workflow.add_edge("response_formatter", END)

# Compile the graph
agent_app = workflow.compile()

print("✅ LangGraph Agent Compiled Successfully")

# ============================================================================
# Example Usage (Use 'python agent.py' to test independently)
# ============================================================================
if __name__ == "__main__":
    import asyncio
    
    async def run_demo():
        # define initial state
        initial_state = {
            "messages": [{"role": "user", "content": "What is the latest news about AI?"}],
            "request_id": "demo-tools",
            "tools_used": [],
            "conversation_context": {}
        }
        
        print("\n🏁 Starting Agent with Web Search Query...")
        result = await agent_app.ainvoke(initial_state)
        
        print("\n✅ Final Result:")
        print("-" * 20)
        print(result["final_response"])
        print("-" * 20)
        print(f"Metadata: {result['conversation_context']}")

    asyncio.run(run_demo())
