import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END

from my_agent.utils.state import MainState
from my_agent.utils.tools import scanEbay, tools

from .model import model

logger = logging.getLogger(__name__)

tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


def llm_call(state: MainState):
    """LLM decides whether to call a tool or not"""
    query = state.get("query", "")
    messages = state.get("messages", [])

    logger.info(f"Current query: {query}")
    logger.info(f"Current messages: {messages}")

    if messages and isinstance(messages[-1], HumanMessage):
        query += " " + str(messages[-1].content)

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are an expert Pokemon card finder for eBay.\n"
                            "Your goal is to help users find specific Pokemon cards.\n"
                            f"Your current task is to narrow down the users search query to something that is specific. Please look at the user message and also {query}.\n"
                            "if their query is in the format Card quality (NM, LP, HP, ETC) or Grading Company Abbreviation(PSA, CGC, BGS, ETC)+Number Grade Set Name Card Name Card Number Release Year, ask for the min or max price if not provided then call refineQuery.\n"
                            "The details required for a specific query is the pokemon name, if they are looking for a raw or graded card, if it is graded what grading company they are looking for and a specific or minimum grade, and a minimum or maximum price."
                            "If a user query doesn't provide all of the required details, ask clarifying questions to narrow down the specific card. \n"
                            "When you have collected all the specific details (Pokemon name, Grade/Condition, Set, Price), "
                            "call the 'refineQuery' tool to optimize the search terms."
                        )
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "query": query,
    }


def search_ebay_node(state: MainState):
    """Searches eBay using the filtered queries."""
    queries = state.get("filtered_queries", [])

    # Call scanEbay directly as a tool
    # The tool returns a string, so we wrap it in a ToolMessage (or HumanMessage since it's an internal node acting as a tool)
    results = scanEbay.invoke({"queries": queries})

    return {
        "messages": [
            ToolMessage(content=str(results), tool_call_id="search_ebay_node")
        ],
        # Clear the queue so we don't re-search automatically next time
        "filtered_queries": [],
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

            # Check if observation is a dict (implies state update)
            if isinstance(observation, dict):
                # Update our state updates with any keys that aren't 'messages'
                # We handle 'messages' separately or let them merge if intended,
                # but typically ToolMessage is the main message artifact.
                # However, refineQuery returns 'messages' as HumanMessage list?
                # That might break the ToolMessage pattern if not careful.
                # For now, let's just merge the whole dict into state_updates
                state_updates.update(observation)
                # We still need a string for the ToolMessage content
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
