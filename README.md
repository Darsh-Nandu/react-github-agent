# 🤖 ReAct GitHub Agent

> A production-ready AI agent powered by LangGraph, FastMCP, and dual-memory systems to seamlessly interact with, manage, and develop GitHub repositories.

**ReAct GitHub Agent** is an advanced, autonomous AI tool designed for developers. It leverages a LangGraph `Think → Act → Observe` loop to execute complex software engineering tasks directly on your GitHub repositories. Backed by an extensible FastMCP server and a dual-memory system, it learns your preferences and project context over time.

---

## Features

- **LangGraph ReAct Loop**: Robust cognitive architecture for autonomous problem solving and tool execution.
- **FastMCP Server Integration**: Custom tools via SSE, including safe Python execution, local file utilities, and web searching.
- **Comprehensive GitHub Access**: Read/write files, manage branches, handle commits, review PRs, and triage issues directly through PyGithub.
- **Dual Memory Architecture**: Seamless context management combining session-based short-term memory (LangGraph) and persistent long-term memory (Mem0).
- **FastAPI Backend**: High-performance, asynchronous backend with real-time streaming via Server-Sent Events (SSE).
- **Dark Terminal UI**: A sleek, developer-focused frontend interface.

---

## Quick Start

### 1. Clone and Install
```bash
git clone [https://github.com/yourusername/react-github-agent.git](https://github.com/yourusername/react-github-agent.git)
cd react-github-agent
pip install -r requirements.txt
````

### 2\. Configure Environment

Copy the example environment file and add your credentials.

```bash
cp .env.example .env
```

Edit `.env` and fill in:

  * `ANTHROPIC_API_KEY`: Your Anthropic API key.
  * `GITHUB_TOKEN`: A GitHub Personal Access Token (requires `repo` and `workflow` scopes).
  * `OPENAI_API_KEY`: (Optional) Used to improve Mem0 semantic search embeddings.

### 3\. Start the MCP Server

In your first terminal, launch the FastMCP tool server:

```bash
python mcp_server/server.py
# → Running on http://localhost:8001
```

### 4\. Start the FastAPI Backend

In your second terminal, launch the main agent API:

```bash
python main.py
# → Running on http://localhost:8000
```

### 5\. Access the UI

Navigate to **http://localhost:8000** in your browser to start interacting with the agent.

-----

## Architecture

The system is decoupled into an intelligent FastAPI backend and a specialized FastMCP tool server, ensuring modularity and secure execution.

```text
FastAPI (port 8000)
  └── LangGraph ReAct Agent
        ├── Short-term memory  (LangGraph MemorySaver, per-session)
        ├── Long-term memory   (Mem0, cross-session)
        ├── FastMCP tools      (connected via SSE to port 8001)
        │     ├── run_python
        │     ├── format_python_code / lint_python_code
        │     ├── fetch_url / web_search
        │     ├── read_local_file / write_local_file
        │     ├── summarize_text
        │     ├── parse_json
        │     └── diff_strings
        └── GitHub tools       (PyGithub)
              ├── list_repos / get_file_tree
              ├── read_github_file / write_github_file
              ├── create_branch / list_branches / list_commits
              ├── search_code
              ├── list_issues / create_issue / add_issue_comment / close_issue
              └── list_pull_requests / create_pull_request / get_pr_diff
```

-----

## Memory Management

### Memory Flow Pipeline

The agent combines contextual awareness with persistent knowledge extraction.

```text
User Query
  │
  ├─→ recall_memories(user_id, query)   # Semantic search in Mem0
  ├─→ inject into system prompt         # Provide long-term context
  ├─→ LangGraph ReAct loop runs         # Agent executes multi-step reasoning
  │
  └─→ save_memory(user_id, facts)       # Heuristic extraction → Saves to Mem0
```
-----

## Frontend UI
<img width="1907" height="918" alt="image" src="https://github.com/user-attachments/assets/54d57d93-dd38-4592-8e1a-8388bef65354" />

## Roadmap & Future Improvements

We are continually refining the agent to be more autonomous, accurate, and user-friendly. Planned updates include:

  * **System Prompt Optimization:** Dynamic injection of repository-specific coding guidelines, architectural constraints, and refined tool-usage rules to reduce hallucinations and improve direct output quality.
  * **Memory Enhancements:** Upgrading memory management with automated decay for outdated facts, vector-based deduplication, and smarter heuristic extraction for Mem0 to prevent context bloat.
  * **UI Output Formatting:** Improving the Dark Terminal UI to support streaming Markdown parsing, syntax-highlighted code blocks with one-click copy, and collapsible tool-execution logs (Think/Act/Observe traces) for a cleaner chat experience.

-----

## 📄 License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).
