import base64
import os

import requests
from langgraph.graph import END, START, StateGraph

from my_agent.utils.nodes import llm_call, search_ebay_node, should_continue, tool_node
from my_agent.utils.state import MainState




def route_after_tool(state: MainState):
    if state.get("can_search", False) and state.get("queries"):
        return "search_ebay_node"
    else:
        return "llm_call"


def create_agent_builder() -> StateGraph:
    """Build the agent graph. Compile a fresh instance for each test run."""

    agent_builder = StateGraph(MainState)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_node("search_ebay_node", search_ebay_node)

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("tool_node", route_after_tool, ["search_ebay_node", "llm_call"])
    agent_builder.add_edge("llm_call", END)
    agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])


    return agent_builder


agent_builder = create_agent_builder()
graph = agent_builder.compile()

if __name__ == "__main__":
    import logging
    import os

    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import MemorySaver

    log_level = (
        logging.DEBUG if os.getenv("AGENT_VERBOSE") else logging.WARNING
    )
    logging.basicConfig(level=log_level)
    for noisy_logger in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Set up persistence for the CLI session

    memory = MemorySaver()

    # Compile a new graph instance with the checkpointer

    # We use the existing agent_builder defined in this file

    cli_graph = agent_builder.compile(checkpointer=memory)

    # Config with a static thread_id for this session

    config = {"configurable": {"thread_id": "1"}}

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
                reply = next(
                    (
                        m.content
                        for m in reversed(result["messages"])
                        if isinstance(m, AIMessage) and m.content
                    ),
                    None,
                )
                print(reply or result["messages"][-1].content)
            
                print("queries: ", result["queries"])
                print("query: ", result["query"])
            else:
                print("Error: ", result)

        except Exception as e:
            print(f"Error: {e}")

            print(
                "Tip: Check if 'MainState' in 'utils/state.py' includes a 'messages' key."
            )
