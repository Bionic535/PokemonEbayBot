from langgraph.graph import END, START, StateGraph

from my_agent.utils.nodes import llm_call, should_continue, tool_node
from my_agent.utils.state import MessagesState

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
graph = agent_builder.compile()

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    print("--- Running Agent Check ---")
    inputs = {"messages": [HumanMessage(content="What is 15 multiplied by 4?")]}

    # Invoke the graph
    result = graph.invoke(inputs)

    # Print the conversation history
    print("\nResult:")
    for m in result["messages"]:
        m.pretty_print()
