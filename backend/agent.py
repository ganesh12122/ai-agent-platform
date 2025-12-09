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
from database import format_context_for_prompt, save_message

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
    conversation_context: Dict[str, Any]  # Previous context (NEW)
    conversation_id: str                  # Which conversation (NEW)
    conversation_turn: int                # Which turn (NEW)
    request_id: str                       # Unique ID for tracing
    final_response: str                   # The generated response to return

# ============================================================================
# 2. DEFINE NODES
# ============================================================================
# Nodes are functions that process the state and return updates.

async def query_analyzer_node(state: AgentState) -> Dict[str, Any]:
    """
    Node A: Query Analyzer
    Analyzes the latest user message, considering conversation history
    """
    
    print("--- 🔍 Analyzing Query ---")
    
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "general"}
    
    last_message = messages[-1]["content"].lower()
    
    # Check for context-dependent intents
    conversation_context = state.get("conversation_context", {})
    previous_intent = None
    
    if conversation_context and "messages" in conversation_context:
        # Look at previous intents from last 3 messages
        # Note: 'messages' in context are chronological, so we look at the end
        recent_msgs = conversation_context["messages"]
        for msg in recent_msgs[-3:]: 
            if msg.get("intent"):
                previous_intent = msg["intent"]
        print(f"  → Context: Previous intent was '{previous_intent}'")
    else:
        print("  → Context: No previous context found")
    
    # Keyword-based detection (with context consideration)
    code_keywords = ["code", "python", "function", "bug", "error", "api", "json", "debug",
                     "write", "implement", "fix", "script", "execute", "run"]
    
    if any(keyword in last_message for keyword in code_keywords):
        intent = "code"
        print("  → Decision: Keywords detected")
    elif previous_intent == "code" and any(word in last_message for word in ["explain", "how", "why", "more", "detail"]):
        # Follow-up to code context
        intent = "code"
        print("  → Decision: Follow-up to previous code context")
    else:
        intent = "general"
        print("  → Decision: General intent")
        
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
    prompt = ""
    
    # ENHANCED: Add conversation context
    conversation_context = state.get("conversation_context", {})
    if conversation_context and conversation_context.get("recent_turns", 0) > 0:
        context_str = await format_context_for_prompt(conversation_context)
        prompt = context_str + "\n\nNow, respond to the user:\n"
    
    # Add System Prompt based on intent to improve generation quality
    intent = state.get("intent", "general")
    if intent == "code":
        prompt += (
            "SYSTEM: You are an expert Python coder. "
            "Write a SINGLE, COMPLETE, and RUNNABLE Python script to solve the user's problem. "
            "The script MUST print the final result to stdout. "
            "Do not use input(). Do not use placeholders. "
            "Ensure all code is inside a single ```python block.\n\n"
        )
    
    prompt += "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    
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

async def memory_update_node(state: AgentState) -> Dict[str, Any]:
    """
    Node H: Memory Update
    Saves the current turn to database for persistence
    """
    
    print("--- 💾 Saving to Memory ---")
    
    conversation_id = state.get("conversation_id")
    if not conversation_id:
        print("  ℹ️ No conversation ID, skipping memory save")
        return {}
    
    # Save user message (if not already saved)
    messages = state.get("messages", [])
    if messages and messages[-1]["role"] == "user":
        user_content = messages[-1]["content"]
        await save_message(
            conversation_id,
            role="user",
            content=user_content,
            intent=state.get("intent")
        )
    
    # Save assistant response
    response = state.get("final_response", "")
    await save_message(
        conversation_id,
        role="assistant",
        content=response,
        model_used=state.get("model"),
        tokens_used=len(response.split())  # Approximate token count
    )
    
    current_tools = state.get("tools_used", [])
    
    return {
        "tools_used": current_tools + ["memory_update"]
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

import re

async def code_runner_node(state: AgentState) -> Dict[str, Any]:
    """
    Node G: Code Runner
    
    Parses the LLM response for Python code blocks and executes them.
    Appends the output to the final response.
    """
    print("--- 🏃 Running Generated Code ---")
    
    intent = state.get("intent", "general")
    tool_calls = state.get("tool_calls", [])
    response_text = state.get("final_response", "")
    
    # Only run if code intent or explicitly requested
    if intent != "code" and "code_executor" not in tool_calls:
        return {"tools_used": state.get("tools_used", []) + ["code_runner"]}
        
    # Extract code blocks
    code_blocks = re.findall(r"```python(.*?)```", response_text, re.DOTALL)
    
    if not code_blocks:
        print("  → No code blocks found to execute")
        return {"tools_used": state.get("tools_used", []) + ["code_runner"]}
        
    print(f"  → Found {len(code_blocks)} code blocks")
    execution_output = ""
    
    for i, code in enumerate(code_blocks):
        print(f"  → Executing block {i+1}...")
        try:
            # Use the existing execute_tool function
            result = await execute_tool("code_executor", code=code.strip())
            output = result.get("result", "")
            execution_output += f"\n\n[Block {i+1} Execution Result]:\n{output}"
        except Exception as e:
            execution_output += f"\n\n[Block {i+1} Error]:\n{str(e)}"
            
    # Append execution results to the final response
    updated_response = response_text + "\n" + execution_output
    
    current_tools = state.get("tools_used", [])
    return {
        "final_response": updated_response,
        "tools_used": current_tools + ["code_runner"]
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
workflow.add_node("code_runner", code_runner_node)
workflow.add_node("response_formatter", response_formatter_node)
workflow.add_node("memory_update", memory_update_node)

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

# Model Executor -> Code Runner (NEW)
workflow.add_edge("model_executor", "code_runner")

# Code Runner -> Formatter
workflow.add_edge("code_runner", "response_formatter")

# Formatter -> Memory Update
workflow.add_edge("response_formatter", "memory_update")

# Memory Update -> End
workflow.add_edge("memory_update", END)

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
