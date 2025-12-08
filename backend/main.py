from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import uuid
from agent import agent_app, AgentState
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import asyncio
import json
from typing import Optional, List, AsyncGenerator
import os
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events"""
    # Startup
    print("=" * 60)
    print("🚀 AI Agent Backend Starting...")
    print("=" * 60)
    health = await check_ollama_health()
    if health:
        print("✅ Ollama connected")
        models = await get_available_models()
        print(f"📦 Loaded models: {models}")
    else:
        print("⚠️  Ollama not responding - make sure: ollama serve is running")
    yield
    # Shutdown
    print("🛑 AI Agent Backend Stopping...")

app = FastAPI(
    title="AI Agent Platform",
    description="Multi-model AI agent with LangGraph orchestration",
    version="0.1.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment Variables
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "mistral"

# ============================================================================
# MODELS & SCHEMAS
# ============================================================================

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = DEFAULT_MODEL
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = True

class ModelInfo(BaseModel):
    name: str
    size: str
    context_size: int
    specialty: str

# ============================================================================
# MODEL REGISTRY (RTX 3050 4GB COMPATIBLE)
# ============================================================================

AVAILABLE_MODELS = {
    "mistral": ModelInfo(
        name="mistral",
        size="7B",
        context_size=32768,
        specialty="General-purpose, fastest"
    ),
    "deepseek-coder": ModelInfo(
        name="deepseek-coder:7b",
        size="7B",
        context_size=4096,
        specialty="Code generation & understanding"
    ),
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def check_ollama_health() -> bool:
    """Check if Ollama server is running"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            return response.status_code == 200
    except Exception as e:
        print(f"❌ Ollama health check failed: {e}")
        return False

async def get_available_models() -> List[str]:
    """Get list of models currently in Ollama"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            data = response.json()
            return [m['name'].split(':')[0] for m in data.get('models', [])]
    except Exception:
        return []

async def query_ollama_stream(
    model: str,
    messages: List[ChatMessage],
    temperature: float = 0.7,
    max_tokens: int = 2048
) -> AsyncGenerator[str, None]:
    """Stream response from Ollama API"""
    
    # Convert messages to prompt format
    prompt = "\n".join([f"{m.role}: {m.content}" for m in messages])
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "temperature": temperature,
        "num_predict": max_tokens,
    }
    
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", f"{OLLAMA_HOST}/api/generate", json=payload) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if 'response' in data:
                            yield data['response']
    except Exception as e:
        yield f"\n[ERROR] {str(e)}"

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Check if backend and Ollama are healthy"""
    ollama_alive = await check_ollama_health()
    return {
        "status": "healthy" if ollama_alive else "unhealthy",
        "ollama": "connected" if ollama_alive else "disconnected",
        "backend": "running",
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }

@app.get("/api/models")
async def list_models():
    """List available models with metadata"""
    available = await get_available_models()
    return {
        "registered": AVAILABLE_MODELS,
        "loaded": available,
        "warning": "RTX 3050 4GB: Only one model at a time."
    }

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint using LangGraph Agent.
    Invokes the agent workflow and streams the response.
    """
    
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    # 1. Prepare Agent State
    # We initialize the state with the user's input and context
    initial_state: AgentState = {
        "messages": [m.dict() for m in request.messages],
        "model": request.model,  # Optional: User can suggest, but agent might override in 'model_selector'
        "intent": "general",     # Default
        "tools_used": [],
        "conversation_context": {},
        "request_id": request_id,
        "final_response": ""
    }
    
    print(f"🔄 invoking agent for request {request_id}")
    
    # 2. Invoke Agent
    # We use ainvoke to run the graph asynchronously.
    try:
        # Note: ainvoke waits for the entire graph to finish.
        # For true streaming of tokens from the LLM, we'd need to stream events from the graph.
        # For now, we simulate streaming the final result to keep frontend compatibility.
        final_state = await agent_app.ainvoke(initial_state)
        
        response_text = final_state.get("final_response", "")
        tools_used = final_state.get("tools_used", [])
        model_used = final_state.get("model", "unknown")
        
    except Exception as e:
        print(f"❌ Agent invocation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # 3. Stream Response
    # We yield the payload in SSE format as expected by the frontend/client
    async def generate():
        # Yield metadata first (optional, or just log it)
        # yield f"data: {json.dumps({'type': 'metadata', 'tools': tools_used})}\n\n"
        
        # Yield the response content
        # Splitting by words to simulate streaming if it's a bulk response
        # or just yielding the whole thing if okay. 
        # Let's yield it in chunks to mimic the "token" structure the client expects.
        
        chunk_size = 10
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i+chunk_size]
            yield f"data: {json.dumps({'token': chunk})}\n\n"
            await asyncio.sleep(0.01) # Small delay to simulate stream
            
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-Model-Used": model_used,
            "X-Tools-Used": ",".join(tools_used)
        }
    )

class QueryRequest(BaseModel):
    query: str

@app.post("/api/route-query")
async def route_query(request: QueryRequest):
    """
    Analyze query using LangGraph Agent infrastructure.
    Invokes the agent to determine intent and model.
    """
    request_id = str(uuid.uuid4())
    
    # 1. Prepare State with dummy message derived from query
    initial_state: AgentState = {
        "messages": [{"role": "user", "content": request.query}],
        "model": "mistral", # Default
        "intent": "general",
        "tools_used": [],
        "conversation_context": {"dry_run": True}, # Hint context
        "request_id": request_id,
        "final_response": ""
    }
    
    try:
        # 2. Invoke Agent
        final_state = await agent_app.ainvoke(initial_state)
        
        # 3. Extract Info
        return {
            "query": request.query,
            "recommended_model": final_state.get("model"),
            "reason": f"Detected intent: {final_state.get('intent')}",
            "available_models": list(AVAILABLE_MODELS.keys()),
            "agent_trace": final_state.get("tools_used")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    print("\n🎯 Starting FastAPI server on http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
