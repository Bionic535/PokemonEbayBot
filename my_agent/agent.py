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

agent_builder.add_edge("search_ebay_node", "llm_call")

# Compile the agent
graph = agent_builder.compile()

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    print("--- Pokemon eBay Agent ---")
    print("Type 'q', 'quit', or 'exit' to end the session.")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["q", "quit", "exit"]:
            print("Goodbye!")
            break

        inputs = {"messages": [HumanMessage(content=user_input)]}

        try:
            # Invoke the graph
            # check pointer=True allows usage of threads if configured, but simple invoke is fine here.
            result = graph.invoke(inputs)

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
