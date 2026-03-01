from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from dotenv import load_dotenv
load_dotenv()

# 1. Graph State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# 2. Tool
import subprocess
from langchain_core.tools import tool


@tool
def walk() -> str:
    """Make the robot walk forward."""

    print("Robot is walking")

    cmd = [
        "ros2",
        "topic",
        "pub",
        "-1",  # publish once
        "/cmd_vel",   # change to /cmd_vel for real robot
        "geometry_msgs/msg/Twist",
        "{linear: {x: 1, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
    ]

    subprocess.run(cmd, check=True)

    return "Walking executed"

tools = [walk]


# 3. LLM with tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

tool_node = ToolNode(tools)


# 4. Agent Node
def agent_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# 5. Router (decides tool call)
def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    # If LLM requested tool
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return END


# 6. Build Graph
builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.set_entry_point("agent")

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)

builder.add_edge("tools", "agent")

graph = builder.compile()

# 7. Run Interactive Loop
if __name__ == "__main__":

    print("Agent ready! Type 'exit' to quit.\n")

    state = {"messages": []}  # persistent conversation memory

    while True:
        user_input = input("You: ")

        # exit condition
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        # add user message
        state["messages"].append(HumanMessage(content=user_input))

        # run graph
        state = graph.invoke(state)

        # print latest agent response
        print("\nAgent:", state["messages"][-1].content)