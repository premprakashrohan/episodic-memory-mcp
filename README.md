# episodic-memory-mcp
An Episodic Memory MCP Server is an application that allows an AI assistant to remember past interactions, retrieve relevant experiences, and use those experiences when answering future questions.
How to create an Episodic Memory MCP Server

An Episodic Memory MCP Server is an application that allows an AI assistant to remember past interactions, retrieve relevant experiences, and use those experiences when answering future questions.

For example, if you tell an AI assistant:

    Monday: “I'm building a React application with FastAPI.”
    Tuesday: “I prefer PostgreSQL for persistence.”
    Friday: “Continue my application from where we left off.”

The assistant can retrieve the earlier conversations and understand that you're working on a React + FastAPI application with PostgreSQL, rather than treating Friday's request as a completely new task.

The best way to build this is to combine MCP (Model Context Protocol), an episodic memory store, semantic search, and an LLM.
1. Understand the architecture

AI client

Claude Desktop, an MCP-compatible client, or your own agent

MCP protocol

Episodic Memory MCP Server

Tools: store episode · recall memories · search · update

Memory database

PostgreSQL + pgvector, or SQLite for a prototype

Retrieval engine

Semantic similarity, metadata filters, recency

Embedding model + optional LLM

Convert episodes to vectors; extract summaries, outcomes, and entities

The key distinction is that episodic memory stores experiences, not just isolated facts. An episode might contain the situation, what happened, the action taken, the outcome, and the timestamp.
2. Choose a technology stack

A practical Python implementation could use the following:
Component	Recommended technology
MCP server	Python MCP SDK (mcp)
API or agent layer	Optional FastAPI
Database	PostgreSQL
Vector search	pgvector
Embeddings	OpenAI embeddings or a local embedding model
Episode summarization	An LLM, optionally
Data validation	Pydantic
Deployment	Docker

For an initial proof of concept, use Python and SQLite. Move to PostgreSQL + pgvector when you need concurrent users, durable storage, metadata filtering, and scalable retrieval.
3. Define the memory model

Each episode should have enough context to help the AI understand what happened.

For example, a stored episode might look like this:

{
  "user_id": "user-123",
  "event": "Chose PostgreSQL for the application",
  "context": "Building a React frontend with FastAPI",
  "action": "Selected PostgreSQL as the primary database",
  "outcome": "Database choice confirmed",
  "tags": ["architecture", "database"],
  "timestamp": "2026-10-10T10:00:00Z"
}

A useful database schema includes:

    id: unique episode identifier.
    user_id: owner of the memory.
    event: what happened.
    context: the situation around the event.
    outcome: what resulted from it.
    created_at: when the episode occurred.
    embedding: optional vector representation for semantic search.
    metadata: project names, tags, importance, and other searchable attributes.

Keep the original episode text even when you generate summaries or embeddings. This lets the assistant retrieve the source context instead of relying only on a compressed summary.
4. Create a working Python MCP server

This starter implementation uses SQLite for durable storage and exposes three MCP tools. It uses keyword matching initially, so you can get the basic application working before adding vector search.

The official MCP Python SDK provides tools through decorators, with input schemas generated from Python type hints.
Step A — Create the project

Install Python 3.10 or newer, then run:

mkdir episodic-memory-mcp
cd episodic-memory-mcp

python -m venv .venv

Activate the virtual environment:

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

Install the MCP SDK using the v1 API for compatibility with the example below:

pip install "mcp[cli]>=1.28,<2"

Note: the official SDK now has a v2 release line, so this example deliberately pins v1. The server API differs between major versions.
Step B — Create server.py

Copy this complete starter server:


import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DB_PATH = Path(__file__).with_name("episodes.db")

mcp = FastMCP("episodic-memory")

def connect_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                event TEXT NOT NULL,
                context TEXT NOT NULL DEFAULT '',
                outcome TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episode_user_date
            ON episodes(user_id, created_at DESC)
        """)

@mcp.tool()
def remember_episode(
    user_id: str,
    event: str,
    context: str = "",
    outcome: str = "",
    tags: str = "",
) -> dict:
    """Store an experience for later recall.

    Tags should be comma-separated. Use a stable user_id for
    the same person across conversations.
    """
    if not user_id.strip() or not event.strip():
        raise ValueError("user_id and event are required")

    episode_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    with connect_db() as conn:
        conn.execute(
            """
            INSERT INTO episodes
            (id, user_id, event, context, outcome, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode_id,
                user_id.strip(),
                event.strip(),
                context.strip(),
                outcome.strip(),
                tags.strip(),
                created_at,
            ),
        )

    return {
        "id": episode_id,
        "stored": True,
        "created_at": created_at,
    }

@mcp.tool()
def recall_episodes(
    user_id: str,
    query: str,
    limit: int = 5,
) -> list[dict]:
    """Find matching past experiences for one user.

    This starter uses keyword matching, not semantic similarity.
    """
    if not user_id.strip():
        raise ValueError("user_id is required")

    limit = max(1, min(limit, 20))
    terms = list(dict.fromkeys(
        word.strip(".,!?;:()[]{}\"'").lower()
        for word in query.split()
        if len(word.strip(".,!?;:()[]{}\"'")) > 1
    ))[:10]

    if not terms:
        return []

    conditions = []
    params = [user_id.strip()]

    for term in terms:
        pattern = f"%{term}%"
        conditions.append(
            "(event LIKE ? OR context LIKE ? "
            "OR outcome LIKE ? OR tags LIKE ?)"
        )
        params.extend([pattern] * 4)

    params.append(limit)

    sql = f"""
        SELECT id, event, context, outcome, tags, created_at
        FROM episodes
        WHERE user_id = ?
          AND ({" OR ".join(conditions)})
        ORDER BY created_at DESC
        LIMIT ?
    """

    with connect_db() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [dict(row) for row in rows]

@mcp.tool()
def recent_episodes(user_id: str, limit: int = 10) -> list[dict]:
    """List the most recent experiences belonging to a user."""
    if not user_id.strip():
        raise ValueError("user_id is required")

    limit = max(1, min(limit, 50))

    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT id, event, context, outcome, tags, created_at
            FROM episodes
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id.strip(), limit),
        ).fetchall()

    return [dict(row) for row in rows]

if __name__ == "__main__":
    initialize_db()
    mcp.run()

Step C — Start the server

python server.py

The server runs over standard input/output (stdio), which is suitable for a local MCP client that launches the server as a subprocess.

Your project now has these tools:
Tool	Purpose
remember_episode	Save a new experience
recall_episodes	Retrieve matching experiences
recent_episodes	Retrieve recent experiences

The SQLite database file, episodes.db, is created in the project directory when the server starts.
5. Test the MCP server

The easiest way to test the tools is with the MCP Inspector, which lets you inspect the server's capabilities and invoke tools interactively.

In your project directory, run:

mcp dev server.py

Follow the URL printed by the command to open the Inspector.

Try the following sequence.

1. Store an episode

Tool: remember_episode

{
  "user_id": "demo-user",
  "event": "Selected PostgreSQL",
  "context": "Building a React and FastAPI app",
  "outcome": "Database decision finalized",
  "tags": "architecture,database"
}

2. Recall an episode

Tool: recall_episodes

{
  "user_id": "demo-user",
  "query": "database architecture",
  "limit": 5
}

Expected result: the previously stored PostgreSQL experience.

3. View recent experiences

Tool: recent_episodes

{
  "user_id": "demo-user",
  "limit": 10
}

6. Upgrade it to genuine episodic memory

The starter server stores experiences persistently, but it is not yet a full semantic episodic memory system. To make it useful for an AI agent across long conversations, add the following capabilities.

A. Semantic retrieval

Convert each episode into an embedding vector. Search by meaning rather than exact words. For example, a query about “the database we settled on” should retrieve an episode mentioning PostgreSQL even if the word “database” was not used in the original event.

B. Episode extraction

After a conversation, use an LLM to identify important events, decisions, user preferences, problems, actions, and outcomes. Store the structured episode rather than blindly saving every message.

C. Contextual recall

Retrieve relevant episodes before the AI answers a new question. Rank them using semantic similarity, recency, importance, and project relevance, then return the most useful memories to the agent.

D. Memory management

Add tools to update, delete, consolidate, and inspect memories. Enforce ownership and authorization in the server rather than trusting a model-supplied user_id.
Recommended production architecture

Conversation / agent

Episode extraction and summarization

MCP memory tools

Store · Search · Recall · Update · Delete

PostgreSQL

Episodes and metadata

pgvector

Embedding similarity

Add authentication, access controls, retention policies, and observability around the server.
7. What should the final MCP interface expose?

I would aim for the following tool API:
MCP tool	Function
store_episode	Persist a structured experience
search_episodes	Search by meaning, date, and tags
recall_context	Retrieve context relevant to the current task
get_episode	Fetch one episode by ID
update_episode	Correct or enrich an existing memory
delete_episode	Delete a memory
consolidate_memories	Merge duplicates and summarize recurring experiences

A useful distinction is that MCP is the interface, not the memory algorithm. Your MCP server exposes the operations; your database and retrieval logic determine how memories are stored and recalled.
8. Choose how you want to build the complete application

I can help you implement the next version end to end. Which direction are you aiming for?

Implementation preferences

Choose what fits your project. You can leave the defaults unchanged.

1. Programming language

Python — recommended for an AI memory prototype

TypeScript / Node.js

2. Memory storage

SQLite — local prototype

PostgreSQL + pgvector — production-oriented

Qdrant — vector database focused

3. Memory retrieval

Keyword search only

Semantic vector search

Hybrid search + recency + importance ranking

4. AI integration

Claude Desktop / compatible desktop MCP client

Custom AI agent or chatbot

REST API + MCP server

[Build my implementation]
