from typing import Annotated, Sequence, TypedDict, Literal
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

load_dotenv()

# Global document state
document_content = " "

# ===== RAG SETUP =====
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
pdf_path = "email_etiquette.pdf"  # Your PDF
chroma_db_path = "./chroma_db"  # Where Chroma stores the vector database

# Load and create vector store
if os.path.exists(pdf_path):
    # Check if Chroma DB already exists
    if os.path.exists(chroma_db_path):
        # Load existing vector store
        vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
        print(f"Loaded existing Chroma database from {chroma_db_path}")
    else:
        # Create new vector store from PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            chunks, 
            embeddings,
            persist_directory=chroma_db_path
        )
        print(f"Created new Chroma database at {chroma_db_path}")
    retriever = vectorstore.as_retriever()
else:
    retriever = None
    print(f"PDF not found: {pdf_path}")

# ===== TOOLS =====
@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base (PDFs) for information about writing, style, or formatting."""
    if retriever is None:
        return "No knowledge base available."
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

@tool
def update_document(content: str) -> str:
    """Update the document content with new content."""
    global document_content
    document_content = content
    return f"Document updated: {document_content[:100]}..."

@tool
def save_document(filename: str) -> str:
    """Save the current document to a file."""
    global document_content
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"
    drafts = "drafts"
    os.makedirs(drafts, exist_ok=True)
    filepath = os.path.join(drafts, filename)
    try:
        with open(filepath, "w") as f:
            f.write(document_content)
        return f"Document saved to {filepath}"
    except Exception as e:
        return f"Error: {str(e)}"

# ===== AGENT STATE =====
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_agent: str

# ===== AGENTS =====
rag_tools = [search_knowledge_base]
drafting_tools = [update_document, save_document]

rag_model = ChatOpenAI(model="gpt-4o-mini").bind_tools(rag_tools)
drafting_model = ChatOpenAI(model="gpt-4o-mini").bind_tools(drafting_tools)
supervisor_model = ChatOpenAI(model="gpt-4o-mini")

def rag_agent(state: AgentState) -> AgentState:
    """Agent that searches knowledge base for writing guidance."""
    system_prompt = SystemMessage(content="""
    You are a research assistant. Your job is to search the knowledge base (PDFs) 
    for information about writing style, formatting, etiquette, or guidelines.
    When asked a question, search the knowledge base and return relevant information.
    """)
    
    messages = [system_prompt] + list(state["messages"])
    response = rag_model.invoke(messages)
    return {"messages": [response], "next_agent": state.get("next_agent", "drafting_agent")}

def drafting_agent(state: AgentState) -> AgentState:
    """Agent that writes and updates documents."""
    system_prompt = SystemMessage(content=f"""
    You are a document writer. You can update and save documents.
    Current document content: {document_content}
    Use the information provided to write high-quality documents.
    """)
    
    messages = [system_prompt] + list(state["messages"])
    response = drafting_model.invoke(messages)
    return {"messages": [response], "next_agent": state.get("next_agent", "drafting_agent")}

def supervisor(state: AgentState) -> AgentState:
    """Boss agent that routes to the right agent."""
    messages = state["messages"]
    if not messages:
        return {"messages": messages, "next_agent": "drafting_agent"}
    
    # Check if document was saved (end condition)
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and 
            "document" in message.content.lower()):
            return {"messages": messages, "next_agent": "end"}
    
    last_message = messages[-1]
    
    # Simple routing logic (you can make this smarter with LLM)
    if isinstance(last_message, HumanMessage):
        content = last_message.content.lower()
        if any(word in content for word in ["search", "find", "lookup", "what", "how", "guideline", "rule"]):
            return {"messages": messages, "next_agent": "rag_agent"}
        else:
            return {"messages": messages, "next_agent": "drafting_agent"}
    
    # After tool execution, check if we should continue or end
    # If last message was a tool result, go back to drafting to process it
    if isinstance(last_message, ToolMessage):
        return {"messages": messages, "next_agent": "drafting_agent"}
    
    return {"messages": messages, "next_agent": "drafting_agent"}

# ===== GRAPH =====
graph = StateGraph(AgentState)

graph.add_node("supervisor", supervisor)
graph.add_node("rag_agent", rag_agent)
graph.add_node("rag_tools", ToolNode(rag_tools))
graph.add_node("drafting_agent", drafting_agent)
graph.add_node("drafting_tools", ToolNode(drafting_tools))

graph.set_entry_point("supervisor")

graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next_agent"],
    {
        "rag_agent": "rag_agent",
        "drafting_agent": "drafting_agent",
        "end": END
    }
)

graph.add_edge("rag_agent", "rag_tools")
graph.add_edge("rag_tools", "supervisor")
graph.add_edge("drafting_agent", "drafting_tools")
graph.add_edge("drafting_tools", "supervisor")

app = graph.compile()


# Track printed messages to avoid duplicates
_printed_message_count = 0

def print_messages(messages: Sequence[BaseMessage]):
    """Function to print messages in a more readable format"""
    global _printed_message_count
    if not messages:
        return
    
    # Only print new messages we haven't seen before
    new_messages = messages[_printed_message_count:]
    for message in new_messages:
        if isinstance(message, HumanMessage):
            print(f"\n👤 USER: {message.content}")
        elif isinstance(message, AIMessage):
            if message.content:
                print(f"\n🤖 AI: {message.content}")
            if hasattr(message, "tool_calls") and message.tool_calls:
                print(f"   🔧 Using tools: {[tc['name'] for tc in message.tool_calls]}")
        elif isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")
    
    _printed_message_count = len(messages)


def run_multi_agent_drafter():
    global _printed_message_count
    _printed_message_count = 0  # Reset counter for new session
    print("\n ===== MULTI-AGENT DRAFTER =====")
    state = {"messages": [], "next_agent": "drafting_agent"}
    
    # Get initial user input
    user_input = input("\nWhat would you like to do? ")
    state["messages"].append(HumanMessage(content=user_input))
    
    # Stream the agent execution
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
        
        # Check if document was saved (end condition)
        messages = step.get("messages", [])
        for message in reversed(messages):
            if (isinstance(message, ToolMessage) and 
                "saved" in message.content.lower() and 
                "document" in message.content.lower()):
                print("\n ===== DRAFTER FINISHED =====")
                return
    
    # Continue conversation loop
    while True:
        _printed_message_count = len(state["messages"])  # Reset to current message count
        user_input = input("\nWhat would you like to do next? (or 'quit' to exit): ")
        if user_input.lower() in ['quit', 'exit', 'done']:
            print("\n ===== DRAFTER FINISHED =====")
            break
        
        state["messages"].append(HumanMessage(content=user_input))
        
        for step in app.stream(state, stream_mode="values"):
            if "messages" in step:
                print_messages(step["messages"])
            
            # Check if document was saved
            messages = step.get("messages", [])
            for message in reversed(messages):
                if (isinstance(message, ToolMessage) and 
                    "saved" in message.content.lower() and 
                    "document" in message.content.lower()):
                    print("\n ===== DRAFTER FINISHED =====")
                    return
            
            # Update state for next iteration
            state = step


if __name__ == "__main__":
    run_multi_agent_drafter()
