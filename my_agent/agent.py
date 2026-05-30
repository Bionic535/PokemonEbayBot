import base64
import os

import requests
from langgraph.graph import END, START, StateGraph

from my_agent.utils.nodes import llm_call, search_ebay_node, should_continue, tool_node
from my_agent.utils.state import MainState


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


def route_after_tool(state: MainState):
    if state.get("filtered_queries"):
        return "search_ebay_node"
    return "llm_call"


def create_agent_builder() -> StateGraph:
    """Build the agent graph. Compile a fresh instance for each test run."""
    agent_builder = StateGraph(MainState)

    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_node("search_ebay_node", search_ebay_node)

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    agent_builder.add_conditional_edges(
        "tool_node",
        route_after_tool,
        {"search_ebay_node": "search_ebay_node", "llm_call": "llm_call"},
    )

    return agent_builder


agent_builder = create_agent_builder()
graph = agent_builder.compile()

if __name__ == "__main__":
    import logging

    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import MemorySaver

    logging.basicConfig(level=logging.INFO)

    # Set up persistence for the CLI session

    memory = MemorySaver()

    # Compile a new graph instance with the checkpointer

    # We use the existing agent_builder defined in this file

    cli_graph = agent_builder.compile(checkpointer=memory)

    # Config with a static thread_id for this session

    config = {"configurable": {"thread_id": "1"}}

    print(get_ebay_token())
    print("--- Pokemon eBay Agent ---")

    print("Type 'q', 'quit', or 'exit' to end the session.")
    print("1. **Which Pokémon card are you looking for?** (e.g., “Charizard,” “Pikachu,” a specific set like *Base Set* or *XYZ*, etc.)")
    print("2. **Do you want a raw (un‑graded) card or a graded one?**")
    print("3. **If you want it graded, which grading company?** (PSA, CGC, BGS, etc.)")
    print("4. **What grade or condition are you targeting?** (e.g., PSA 9, CGC 8, “NM‑Mint,” “Near Mint,” etc.)")
    print("5. **Do you have a price range in mind?** (minimum, maximum, or a specific price?)")

    while True:
        user_input = input("\nUser: ")

        if user_input.lower() in ["q", "quit", "exit"]:
            print("Goodbye!")

            break

        # Only pass the new message. 'query' will be retrieved from the persisted state.

        inputs = {"messages": [HumanMessage(content=user_input)]}

        try:
            # Invoke the graph using the CLI-specific graph and config

            result = cli_graph.invoke(inputs, config=config)

            print("Agent:", end=" ")

            # Attempt to print the last message content

            if "messages" in result and result["messages"]:
                print(result["messages"][-1].content)

            else:
                print(result)

        except Exception as e:
            print(f"Error: {e}")

            print(
                "Tip: Check if 'MainState' in 'utils/state.py' includes a 'messages' key."
            )
