# LangGraph Multi-Agent Drafter

An AI-powered document drafting system built with LangGraph and LangChain. This multi-agent system combines RAG (Retrieval Augmented Generation) with document drafting capabilities to create intelligent, context-aware documents.

## Features

- 🤖 **Multi-Agent Architecture**: Supervisor agent orchestrates specialized RAG and drafting agents
- 📚 **RAG Integration**: Searches PDF knowledge base for writing guidelines and best practices
- 📝 **Intelligent Drafting**: Uses AI to create and edit documents with context from knowledge base
- 💾 **Auto-Save**: Automatically saves documents to the `drafts/` folder
- 🔄 **State Management**: Uses LangGraph to manage conversation flow and agent coordination
- 🛠️ **Tool Integration**: Built-in tools for knowledge search, document updates, and saving

## Architecture

The system uses a **supervisor pattern** with three specialized agents:

1. **Supervisor Agent**: Routes tasks to the appropriate agent based on user input
2. **RAG Agent**: Searches PDF knowledge base (email_etiquette.pdf) for writing guidelines
3. **Drafting Agent**: Writes and updates documents using AI and retrieved knowledge

## Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. Clone this repository or navigate to the project directory:
```bash
cd langgraph-drafter
```

2. Install the required dependencies:
```bash
pip install -r requirement.txt
```

## Setup

1. Create a `.env` file in the project root directory:
```bash
touch .env
```

2. Add your OpenAI API key to the `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

3. Place your PDF knowledge base file:
   - Name it `email_etiquette.pdf` (or update the path in `multi_agent_drafter.py`)
   - The system will automatically create a Chroma vector database on first run

## Usage

### Multi-Agent System (Recommended)

Run the multi-agent drafter:
```bash
python multi_agent_drafter.py
```

The system will:
1. Load or create the Chroma vector database from your PDF
2. Start with a supervisor that routes your requests
3. Use RAG agent when you ask about guidelines/rules
4. Use drafting agent to write and update documents
5. Automatically save documents when finished

### Simple Single-Agent System

For a simpler version without RAG:
```bash
python drafter.py
```

## How It Works

### Multi-Agent Flow

1. **User Input** → Supervisor analyzes the request
2. **Routing Decision**:
   - Questions about guidelines/rules → RAG Agent
   - Document writing/editing → Drafting Agent
3. **RAG Agent**: Searches PDF knowledge base and returns relevant information
4. **Drafting Agent**: Uses retrieved knowledge to write high-quality documents
5. **Tools Execution**: Updates document or saves to file
6. **Completion**: Stops when document is saved

### Tools

- **`search_knowledge_base(query: str)`**: Searches the PDF knowledge base for writing guidelines, style rules, or formatting information
- **`update_document(content: str)`**: Updates the document content with new text
- **`save_document(filename: str)`**: Saves the current document to `drafts/` folder and ends the session

### RAG Setup

- **Vector Database**: Chroma (stored in `./chroma_db/`)
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Chunking**: 250 characters with 50 character overlap (optimized for small PDFs)
- **PDF Loading**: Automatically loads and indexes `email_etiquette.pdf`

### State Management

The system maintains conversation state through:
- Message history (user messages, AI responses, tool results)
- Current document content (stored globally)
- Agent routing decisions (supervisor tracks which agent to use)
- Conditional routing based on whether the document has been saved

## Project Structure

```
langgraph-drafter/
├── multi_agent_drafter.py  # Multi-agent system with RAG (main)
├── drafter.py              # Simple single-agent version
├── rag.py                  # RAG setup code
├── requirement.txt         # Python dependencies
├── README.md              # This file
├── .env                    # Environment variables (create this)
├── email_etiquette.pdf     # Knowledge base PDF
├── chroma_db/              # Vector database (auto-generated)
└── drafts/                 # Saved documents folder
```

## Dependencies

- `python-dotenv`: Loads environment variables from `.env` file
- `langchain-core`: Core LangChain functionality for messages and tools
- `langchain-openai`: OpenAI integration for ChatOpenAI model and embeddings
- `langgraph`: Graph-based state management for agent workflows
- `langchain-chroma`: Chroma vector database integration
- `langchain-text-splitters`: Text chunking for RAG
- `langchain-community`: PDF document loaders

## Example Interaction

```
===== MULTI-AGENT DRAFTER =====

What would you like to do? draft a formal email to my professor

👤 USER: draft a formal email to my professor

🤖 AI: I'll help you draft a formal email. Let me first check the guidelines...

   🔧 Using tools: ['search_knowledge_base']

🛠️ TOOL RESULT: [Guidelines from PDF about formal email structure...]

🤖 AI: Based on the guidelines, here's a formal email draft...

   🔧 Using tools: ['update_document']

🛠️ TOOL RESULT: Document updated: Dear Professor...

🤖 AI: The email has been drafted following formal email guidelines...

   🔧 Using tools: ['save_document']

🛠️ TOOL RESULT: Document saved to drafts/email_to_professor.txt

===== DRAFTER FINISHED =====
```

## Notes

- The system uses GPT-4o-mini for cost-effective document drafting
- Documents are saved as `.txt` files in the `drafts/` folder
- The Chroma database is created automatically on first run and persists between sessions
- The system automatically adds `.txt` extension if not provided
- The session ends automatically when you save the document
- RAG chunk size is optimized for small PDFs (250 chars, 50 overlap)

## License

This project is open source and available for personal and educational use.
