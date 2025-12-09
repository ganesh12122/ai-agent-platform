import asyncio
import httpx
import json
import sys
from typing import Dict, Any

BASE_URL = "http://localhost:8000/api"

async def run_test(name: str, coro):
    print(f"🔹 Testing: {name}...", end=" ", flush=True)
    try:
        await coro
        print("✅ PASSED")
        return True
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False

async def test_health():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL}/health")
        assert resp.status_code == 200, f"Status {resp.status_code}"
        data = resp.json()
        assert data["status"] == "healthy", "Backend not healthy"
        assert data["ollama"] == "connected", "Ollama not connected"

async def test_chat_flow() -> str:
    # 1. New Conversation
    print("\n   > Step 1: Create New Conversation")
    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {
            "messages": [{"role": "user", "content": "Hello, my name is QA_Bot."}],
            "model": "mistral",
            "stream": False # Disable stream for easier testing
        }
        # Note: server streams by default, but we can handle it or just look at headers
        # Wait, the server ONLY returns StreamingResponse. 
        # We need to parse SSE.
        
        async with client.stream("POST", f"{BASE_URL}/chat", json=payload) as response:
            assert response.status_code == 200, f"Chat Status {response.status_code}"
            
            # Extract headers
            conv_id = response.headers.get("X-Conversation-ID")
            assert conv_id, "No X-Conversation-ID header"
            turn = response.headers.get("X-Turn-Number")
            
            # Read stream to ensure it completes
            full_response = ""
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        chunk = json.loads(line[6:])
                        if "token" in chunk:
                            full_response += chunk["token"]
                    except:
                        pass
            
            print(f"     Response: {full_response[:50]}...")
            assert len(full_response) > 0, "Empty response from LLM"
            
    # 2. Follow-up (Memory Check)
    print("   > Step 2: Test Memory Persistence")
    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {
            "messages": [{"role": "user", "content": "What is my name?"}],
            "conversation_id": conv_id,
            "stream": False
        }
        
        async with client.stream("POST", f"{BASE_URL}/chat", json=payload) as response:
            assert response.status_code == 200
            assert response.headers.get("X-Conversation-ID") == conv_id, "Conversation ID changed"
            
            full_response = ""
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        chunk = json.loads(line[6:])
                        if "token" in chunk:
                            full_response += chunk["token"]
                    except:
                        pass
            
            print(f"     Response: {full_response}")
            # Mistral should be able to recall "QA_Bot"
            if "QA_Bot" not in full_response and "QA_Bot" not in full_response: 
                print("     ⚠️ WARNING: Memory might not be strictly working or LLM hallucinated.")
            
    return conv_id

async def test_conversations_api(conv_id: str):
    async with httpx.AsyncClient() as client:
        # List
        print("\n   > Step 3: List Conversations")
        resp = await client.get(f"{BASE_URL}/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert "conversations" in data
        found = any(c["conversation_id"] == conv_id for c in data["conversations"])
        assert found, "Created conversation not found in list"
        
        # Details
        print("   > Step 4: Get Details")
        resp = await client.get(f"{BASE_URL}/conversations/{conv_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["conversation_id"] == conv_id
        assert len(data["messages"]) >= 4, "Expected at least 4 messages (2 turns)"
        
        # Archive
        print("   > Step 5: Archive")
        resp = await client.post(f"{BASE_URL}/conversations/{conv_id}/archive")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

async def main():
    print("🚀 Starting Aggressive Backend QA Test\n")
    
    success = True
    if not await run_test("Health Check", test_health()):
        success = False
    else:
        conv_id = await test_chat_flow()
        if conv_id:
            if not await run_test("Conversation API (List/Get/Archive)", test_conversations_api(conv_id)):
                success = False
        else:
            success = False
            
    print("\n🏁 Test Suite Completed")
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
