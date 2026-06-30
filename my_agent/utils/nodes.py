import base64
import logging
import os
from typing import Literal

import requests
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import END

from my_agent.utils.state import MainState
from my_agent.utils.tools import scanEbay, tools
from .model import model
from my_agent.utils.prompts import build_system_prompt

logger = logging.getLogger(__name__)

tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

def get_ebay_token():
    client_id = os.getenv("EBAY_CLIENT_ID")
    client_secret = os.getenv("EBAY_CLIENT_SECRET")
    environment = os.getenv("EBAY_ENVIRONMENT")
    if not client_id or not client_secret or not environment:
        raise ValueError("EBAY_CLIENT_ID and EBAY_CLIENT_SECRET must be set")

    basic_auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    r = requests.post(
        f"https://api.{environment}.ebay.com/identity/v1/oauth2/token",
        headers={
            "Authorization": f"Basic {basic_auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "client_credentials",
            "scope": "https://api.ebay.com/oauth/api_scope",
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise ValueError(f"Failed to get eBay token: {r.json()}")
    return r.json()["access_token"]
    
    
    

def search(query: list[str]):
    return 0; #TODO: Implement eBay search logic
    
def llm_call(state: MainState):
    """LLM decides whether to call a tool or not"""
    current_query = state.get("query", {})
    messages = state.get("messages", [])

    logger.debug("Current query: %s", current_query)
    logger.debug("Current messages: %s", messages)

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content=build_system_prompt(state)
                    )
                ]
                + messages
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def search_ebay_node(state: MainState):
    """Searches eBay using completed query specs."""
    queries = state.get("queries", [])
    logger.info("Searching eBay with queries: %s", queries)
    results = scanEbay.invoke({"queries": queries})

    return {
        "messages": [
            AIMessage(content=f"I found the following items on eBay: {results}")
        ],
        "queries": [],
        "can_search": False,
    }


def tool_node(state: MainState):
    """Performs the tool call"""
    result = []
    state_updates = {}
    # We look at the last message to find tool calls
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])

            if isinstance(observation, dict):
                # Never merge tool-returned messages; every tool_call needs a ToolMessage.
                state_updates.update(
                    {k: v for k, v in observation.items() if k != "messages"}
                )
                content_str = str(observation)
            else:
                content_str = str(observation)

            result.append(
                ToolMessage(content=content_str, tool_call_id=tool_call["id"])
            )

    # Return messages + any other state updates
    return {"messages": result, **state_updates}


def should_continue(state: MainState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END
