from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import os

load_dotenv()


# Global variable to store document content 
document_content = " "

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update_document(content: str) -> str:
    """Update the document content with the new content."""

    global document_content
    document_content = content
    return f"Document updated with: {document_content}"

@tool
def save_document(filename: str) -> str:
    """Save the current document to a text file and finish the process.

    Args:
        filename: Name of the text file.
    """

    global document_content
 
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    drafts = "drafts"
    os.makedirs(drafts, exist_ok=True)
  
    filepath = os.path.join(drafts, filename)
    
    try:
        with open(filepath, "w") as f:
            f.write(document_content)
        print(f"Document saved to {filepath}")
        return f"Document saved to {filepath}"
    except Exception as e:
        return f"Error saving document: {str(e)}"

tools = [update_document, save_document]

model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are a helpful assistant that can update a document with new content.
    You can use the following tools to update the document:
    - update_document: Update the document content with the new content.
    - save_document: Save the current document to a text file and finish the process.
    Make sure to always show the current document state after modifications.

    The current document content is: {document_content}
    """)

    if not state["messages"]:
        user_input = "I'm ready to start drafting the document."
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("What would you like to do with the document? ")
        print(f"User input: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"Using tools: {[tc['name'] for tc in response.tool_calls]}")
    
    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """Determine if the agent should continue drafting the document."""
    messages = state["messages"]

    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and
            "saved" in message.content.lower() and 
            "document" in message.content.lower()):
            return "end"
    
    return "continue"


def print_messages(messages: Sequence[BaseMessage]):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")    

graph = StateGraph(AgentState)

graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")


graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END
    },
)

app = graph.compile()


def run_document_agent():
    print("\n ===== DRAFTER =====")
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_document_agent()
    