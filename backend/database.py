"""
Conversation Memory System
Handles saving and loading conversations from SQLite using aiosqlite for async support.
"""

import aiosqlite
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from contextlib import asynccontextmanager

# Database path
DB_PATH = Path(__file__).parent / "conversations.db"

# ============================================================================
# 1. DATABASE INITIALIZATION
# ============================================================================

async def init_database():
    """
    Initialize SQLite database with conversation tables
    Run this once at startup
    """
    async with get_db_connection() as conn:
        # Conversations table - stores individual messages
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                turn_number INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model_used TEXT,
                intent TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                tokens_used INTEGER DEFAULT 0,
                UNIQUE(conversation_id, turn_number),
                FOREIGN KEY (conversation_id) 
                    REFERENCES conversation_metadata(conversation_id)
            )
        """)
        
        # Metadata table - stores conversation summaries
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_metadata (
                conversation_id TEXT PRIMARY KEY,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                title TEXT,
                model_preference TEXT DEFAULT 'auto',
                is_archived BOOLEAN DEFAULT 0,
                turn_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0
            )
        """)
        
        # Create indexes for faster queries
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_id 
            ON conversations(conversation_id)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON conversations(timestamp)
        """)
        
        await conn.commit()
        print("✅ Database initialized successfully")


# ============================================================================
# 2. DATABASE CONNECTION
# ============================================================================

@asynccontextmanager
async def get_db_connection():
    """
    Async context manager for database connections
    Ensures proper cleanup and error handling
    """
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = aiosqlite.Row  # Access columns by name
    try:
        yield conn
    finally:
        await conn.close()


# ============================================================================
# 3. CONVERSATION MANAGEMENT
# ============================================================================

async def create_conversation(model_preference: str = "auto") -> str:
    """
    Create a new conversation
    
    Args:
        model_preference: Default model for this conversation
    
    Returns:
        conversation_id (UUID string)
    """
    conversation_id = str(uuid.uuid4())
    
    async with get_db_connection() as conn:
        await conn.execute("""
            INSERT INTO conversation_metadata 
            (conversation_id, model_preference, title)
            VALUES (?, ?, ?)
        """, (conversation_id, model_preference, f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"))
        await conn.commit()
    
    print(f"✅ Created conversation: {conversation_id}")
    return conversation_id


async def save_message(
    conversation_id: str,
    role: str,
    content: str,
    model_used: str = None,
    intent: str = None,
    tokens_used: int = 0
) -> int:
    """
    Save a message to conversation
    
    Returns:
        turn_number of saved message
    """
    
    async with get_db_connection() as conn:
        # Get current turn count
        async with conn.execute("""
            SELECT COUNT(*) FROM conversations 
            WHERE conversation_id = ?
        """, (conversation_id,)) as cursor:
            row = await cursor.fetchone()
            turn_count = row[0]
            
        turn_number = turn_count + 1
        
        # Save message
        await conn.execute("""
            INSERT INTO conversations 
            (conversation_id, turn_number, role, content, 
             model_used, intent, tokens_used)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (conversation_id, turn_number, role, content, 
              model_used, intent, tokens_used))
        
        # Update metadata
        await conn.execute("""
            UPDATE conversation_metadata 
            SET updated_at = CURRENT_TIMESTAMP,
                turn_count = turn_count + 1,
                total_tokens = total_tokens + ?
            WHERE conversation_id = ?
        """, (tokens_used, conversation_id))
        
        await conn.commit()
    
    return turn_number


async def load_conversation(
    conversation_id: str,
    limit: int = None
) -> List[Dict[str, Any]]:
    """
    Load conversation history
    """
    
    async with get_db_connection() as conn:
        if limit:
            async with conn.execute("""
                SELECT role, content, model_used, intent, timestamp, tokens_used
                FROM conversations
                WHERE conversation_id = ?
                ORDER BY turn_number DESC
                LIMIT ?
            """, (conversation_id, limit)) as cursor:
                rows = await cursor.fetchall()
                # Reverse back to chronological order
                messages = [dict(row) for row in rows][::-1]
        else:
            async with conn.execute("""
                SELECT role, content, model_used, intent, timestamp, tokens_used
                FROM conversations
                WHERE conversation_id = ?
                ORDER BY turn_number ASC
            """, (conversation_id,)) as cursor:
                rows = await cursor.fetchall()
                messages = [dict(row) for row in rows]
    
    return messages


async def get_recent_context(
    conversation_id: str,
    num_turns: int = 5
) -> Optional[Dict[str, Any]]:
    """
    Get recent conversation context for injection into prompts
    """
    
    async with get_db_connection() as conn:
        # Get metadata
        async with conn.execute("""
            SELECT created_at, turn_count, model_preference, title
            FROM conversation_metadata
            WHERE conversation_id = ?
        """, (conversation_id,)) as cursor:
            metadata_row = await cursor.fetchone()
            
        if not metadata_row:
            return None
        
        # Get recent messages
        async with conn.execute("""
            SELECT role, content, model_used, intent
            FROM conversations
            WHERE conversation_id = ?
            ORDER BY turn_number DESC
            LIMIT ?
        """, (conversation_id, num_turns)) as cursor:
            rows = await cursor.fetchall()
            messages = [dict(row) for row in rows][::-1]  # Chronological order
    
    return {
        "conversation_id": conversation_id,
        "messages": messages,
        "summary": metadata_row["title"],
        "turn_count": metadata_row["turn_count"],
        "created_at": metadata_row["created_at"],
        "model_preference": metadata_row["model_preference"],
        "recent_turns": len(messages)
    }


async def list_conversations(limit: int = 20) -> List[Dict[str, Any]]:
    """
    List all active conversations
    """
    
    async with get_db_connection() as conn:
        async with conn.execute("""
            SELECT conversation_id, title, created_at, updated_at, 
                   turn_count, model_preference
            FROM conversation_metadata
            WHERE is_archived = 0
            ORDER BY updated_at DESC
            LIMIT ?
        """, (limit,)) as cursor:
            rows = await cursor.fetchall()
            conversations = [dict(row) for row in rows]
    
    return conversations


async def archive_conversation(conversation_id: str) -> bool:
    """
    Archive a conversation (soft delete)
    """
    
    async with get_db_connection() as conn:
        await conn.execute("""
            UPDATE conversation_metadata
            SET is_archived = 1
            WHERE conversation_id = ?
        """, (conversation_id,))
        await conn.commit()
    
    return True


async def get_conversation_stats(conversation_id: str) -> Dict[str, Any]:
    """
    Get statistics about a conversation
    """
    
    async with get_db_connection() as conn:
        # Get basic stats
        async with conn.execute("""
            SELECT COUNT(*) as turn_count,
                   SUM(tokens_used) as total_tokens,
                   GROUP_CONCAT(DISTINCT model_used) as models,
                   GROUP_CONCAT(DISTINCT intent) as intents
            FROM conversations
            WHERE conversation_id = ?
        """, (conversation_id,)) as cursor:
            stats_row = await cursor.fetchone()
        
        # Get metadata for duration
        async with conn.execute("""
            SELECT created_at, updated_at
            FROM conversation_metadata
            WHERE conversation_id = ?
        """, (conversation_id,)) as cursor:
            meta_row = await cursor.fetchone()
    
    if not meta_row:
        return {}
        
    created = datetime.fromisoformat(meta_row["created_at"])
    updated = datetime.fromisoformat(meta_row["updated_at"])
    duration_minutes = int((updated - created).total_seconds() / 60)
    
    return {
        "turn_count": stats_row["turn_count"],
        "total_tokens": stats_row["total_tokens"] or 0,
        "avg_tokens_per_turn": round((stats_row["total_tokens"] or 0) / max(1, stats_row["turn_count"] or 1)),
        "models_used": (stats_row["models"] or "").split(","),
        "intents": (stats_row["intents"] or "").split(","),
        "duration_minutes": duration_minutes
    }


# ============================================================================
# 4. CONTEXT FORMATTING
# ============================================================================

async def format_context_for_prompt(context: Dict[str, Any]) -> str:
    """
    Format conversation context as a string to inject into LLM prompt
    
    Args:
        context: Output from get_recent_context()
    
    Returns:
        Formatted string for injection into prompt
    """
    
    messages = context.get("messages", [])
    summary = context.get("summary", "New conversation")
    
    prompt_context = f"=== CONVERSATION CONTEXT ===\\nSummary: {summary}\\nRecent Exchanges ({len(messages)} messages):\\n\\n"
    
    for msg in messages:
        # Truncate very long messages in context to save tokens
        content = msg['content']
        if len(content) > 100:
            content = content[:100] + "..."
            
        role = msg['role'].upper()
        prompt_context += f"{role}: {content}\\n\\n"
    
    prompt_context += "=== END CONTEXT ===\\n\\nUse this context to maintain conversation continuity."
    
    return prompt_context


# ============================================================================
# 4. TEST / DEMO
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import os
    
    # Ensure event loop for Windows
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    async def test():
        print("Testing Memory System using aiosqlite...\n")
        
        # Initialize database
        await init_database()
        
        # Create conversation
        conv_id = await create_conversation()
        print(f"Created: {conv_id}\n")
        
        # Save messages
        await save_message(conv_id, "user", "What is Python?", intent="general")
        await save_message(conv_id, "assistant", "Python is a programming language...", model_used="mistral")
        await save_message(conv_id, "user", "Give me a code example", intent="code")
        await save_message(conv_id, "assistant", "def hello():\n    print('Hello')", model_used="deepseek-coder")
        
        # Load conversation
        messages = await load_conversation(conv_id)
        print(f"Loaded {len(messages)} messages:")
        for msg in messages:
            print(f"  {msg['role']}: {msg['content'][:50]}...")
        
        # Get context
        context = await get_recent_context(conv_id, num_turns=2)
        print(f"\nContext: {context['turn_count']} turns, Summary: {context['summary']}")
        
        # List conversations
        conversations = await list_conversations()
        print(f"\nTotal conversations: {len(conversations)}")
        
        # Get stats
        stats = await get_conversation_stats(conv_id)
        print(f"\nStats: {stats}")
    
    asyncio.run(test())
