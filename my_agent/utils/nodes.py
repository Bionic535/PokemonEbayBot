import os
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END

from my_agent.utils.state import MessagesState
from my_agent.utils.tools import tools

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    print("\nCRITICAL ERROR: NVIDIA_API_KEY not found in environment variables!")
else:
    print(f"\nDEBUG: NVIDIA_API_KEY loaded. Starts with: {api_key[:4]}...")

# Initialize model explicitly using NVIDIA
model = ChatNVIDIA(
    model="nvidia/nemotron-3-nano-30b-a3b",
    api_key=api_key,
    temperature=1,
    top_p=1,
    max_tokens=16384,
    reasoning_budget=16384,
    chat_template_kwargs={"enable_thinking": True},
)


tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: MessagesState):
    """Performs the tool call"""

    result = []
    # We look at the last message to find tool calls
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
            )

    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END
