from langgraph.graph import END, START, StateGraph

from my_agent.utils.nodes import llm_call, search_ebay_node, should_continue, tool_node
from my_agent.utils.state import MainState


def route_after_tool(state: MainState):
    if state.get("filtered_queries"):
        return "search_ebay_node"
    return "llm_call"


# Build workflow
agent_builder = StateGraph(MainState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("search_ebay_node", search_ebay_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])

# Conditional edge from tool_node -> search_ebay_node OR llm_call
agent_builder.add_conditional_edges(
    "tool_node",
    route_after_tool,
    {"search_ebay_node": "search_ebay_node", "llm_call": "llm_call"},
)


# Compile the agent
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

    print("--- Pokemon eBay Agent ---")

    print("Type 'q', 'quit', or 'exit' to end the session.")

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
