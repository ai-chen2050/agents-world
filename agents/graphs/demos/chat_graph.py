from typing import Annotated
from typing_extensions import TypedDict
import json

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_ollama.llms import OllamaLLM
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition

"""
0. Graph State
    State is a TypedDict that defines the state of the graph.
    It has a messages key that is a list of messages.
    The messages key is annotated with add_messages, which defines how the messages should be updated.
    In this case, it appends messages to the list, rather than overwriting them.
"""
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

"""
1. LLM
    We use OllamaLLM to create a chatbot.
"""
llm = OllamaLLM(model="nezahatkorkmaz/deepseek-v3")
# llm2 = ChatAnthropic(model="claude-3-5-sonnet-20240620")

"""
2. LLM with Tools
    We tell the LLM which tools it can call
"""
# Initialize the DuckDuckGo search tool
search_tool = DuckDuckGoSearchRun(max_results=2)
tools = [search_tool]
# llm_with_tools = llm.bind_tools(tools)


"""
3. Graph Builder
    We use StateGraph to create a graph builder.
"""
graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}
    # return {"messages": [llm_with_tools.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

# Add a transition from the START node to the chatbot node
graph_builder.add_edge(START, "chatbot")

# Add a transition from the chatbot node to the END node
graph_builder.add_edge("chatbot", END)

# or 
# graph_builder.set_entry_point("chatbot")
# graph_builder.set_finish_point("chatbot")


tool_node = ToolNode(tools=[search_tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")

# compile the graph

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


"""
5. Main Loop
    We use the graph to stream updates.
"""
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1])

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break