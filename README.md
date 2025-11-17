# LangGraph Drafter

An AI-powered document drafting agent built with LangGraph and LangChain. This interactive agent helps you draft and edit documents through natural language conversation, automatically saving your work when you're done.

## Features

- 🤖 **AI-Powered Drafting**: Uses OpenAI's GPT-4o-mini model to help you create and edit documents
- 📝 **Interactive Editing**: Update document content through natural language commands
- 💾 **Auto-Save**: Automatically saves documents to text files when you're finished
- 🔄 **State Management**: Uses LangGraph to manage conversation flow and document state
- 🛠️ **Tool Integration**: Built-in tools for updating and saving documents

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

## Usage

Run the drafter agent:
```bash
python drafter.py
```

The agent will:
1. Start with an empty document
2. Prompt you for what you'd like to do with the document
3. Use AI to help draft and edit the content
4. Automatically save the document when you're finished

### Example Interaction

```
===== DRAFTER =====
What would you like to do with the document? Write a letter to Tom

AI: I'll help you write a letter to Tom. Let me draft that for you...

🛠️ TOOL RESULT: Document updated with: Dear Tom, ...

What would you like to do with the document? Make it more formal

AI: I'll make the letter more formal...

🛠️ TOOL RESULT: Document updated with: Dear Mr. Tom, ...

What would you like to do with the document? Save it as letter_to_tom

🛠️ TOOL RESULT: Document saved to letter_to_tom.txt

===== DRAFTER FINISHED =====
```

## How It Works

The drafter uses a **LangGraph state graph** to manage the conversation flow:

1. **Agent Node**: Processes user input and generates AI responses using the OpenAI model
2. **Tools Node**: Executes tool calls (update_document, save_document)
3. **Conditional Logic**: Continues the conversation until the document is saved

### Tools

- **`update_document(content: str)`**: Updates the document content with new text
- **`save_document(filename: str)`**: Saves the current document to a `.txt` file and ends the session

### State Management

The agent maintains conversation state through:
- Message history (user messages, AI responses, tool results)
- Current document content (stored globally)
- Conditional routing based on whether the document has been saved

## Project Structure

```
langgraph-drafter/
├── drafter.py          # Main application file
├── requirement.txt     # Python dependencies
├── README.md          # This file
└── .env               # Environment variables (create this)
```

## Dependencies

- `python-dotenv`: Loads environment variables from `.env` file
- `langchain-core`: Core LangChain functionality for messages and tools
- `langchain-openai`: OpenAI integration for ChatOpenAI model
- `langgraph`: Graph-based state management for agent workflows

## Notes

- The agent uses GPT-4o-mini for cost-effective document drafting
- Documents are saved as `.txt` files in the current directory
- The agent automatically adds `.txt` extension if not provided
- The session ends automatically when you save the document

## License

This project is open source and available for personal and educational use.

