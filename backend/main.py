from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
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
    """Chat endpoint with streaming support"""
    
    # Health check
    if not await check_ollama_health():
        raise HTTPException(status_code=503, detail="Ollama server offline")
    
    # Validate model
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model {request.model} not available"
        )
    
    # Check if model is loaded
    loaded_models = await get_available_models()
    if request.model not in loaded_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model {request.model} not loaded"
        )
    
    async def generate():
        async for token in query_ollama_stream(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        ):
            yield f"data: {json.dumps({'token': token})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Model-Used": request.model}
    )

class QueryRequest(BaseModel):
    query: str

@app.post("/api/route-query")
async def route_query(request: QueryRequest):
    """Analyze query and recommend best model"""
    query_lower = request.query.lower()
    
    if any(keyword in query_lower for keyword in ["code", "debug", "function", "algorithm"]):
        recommended = "deepseek-coder"
        reason = "Code-related query detected"
    else:
        recommended = "mistral"
        reason = "General-purpose query"
    
    return {
        "query": request.query,
        "recommended_model": recommended,
        "reason": reason,
        "available_models": list(AVAILABLE_MODELS.keys())
    }



if __name__ == "__main__":
    import uvicorn
    print("\n🎯 Starting FastAPI server on http://localhost:3000")
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)
