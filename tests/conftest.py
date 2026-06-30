import os

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from my_agent.agent import create_agent_builder
from langgraph.checkpoint.memory import MemorySaver

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: live LLM integration tests")


def collect_visited_nodes(compiled_graph, inputs):
    visited = []
    final_state = None
    queries_at_search = None
    messages = inputs.get("messages", [])
    config = {"configurable": {"thread_id": "1"}}
    for i in messages:
        inputs["messages"] = [i]
        for mode, chunk in compiled_graph.stream(
            inputs, stream_mode=["updates", "values"], config=config
        ):
            if mode == "updates":
                visited.extend(chunk.keys())
                if "search_ebay_node" in chunk and final_state is not None:
                    # values mode hasn't run yet for this step;
                    # queries are still in state before the node clears them
                    queries_at_search = final_state.get("queries", [])
            elif mode == "values":
                final_state = chunk

    return visited, final_state, queries_at_search


def build_inputs(case_input: dict) -> dict:
    """Convert YAML case input into graph invoke inputs."""
    inputs = {}
    if "messages" in case_input:
        inputs["messages"] = [
            HumanMessage(content=msg["content"])
            for msg in case_input["messages"]
            if msg.get("role") == "human"
        ]
    if "query" in case_input:
        inputs["query"] = case_input["query"]
    if "queries" in case_input:
        inputs["queries"] = case_input["queries"]
    if "can_search" in case_input:
        inputs["can_search"] = case_input["can_search"]
    return inputs


@pytest.fixture(scope="session")
def openai_api_key():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set; skipping live LLM integration tests")
    return api_key


@pytest.fixture
def compiled_graph(openai_api_key):
    checkpointer = MemorySaver()
    return create_agent_builder().compile(checkpointer=checkpointer)
