import os

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from my_agent.agent import create_agent_builder


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: live LLM integration tests")


def collect_visited_nodes(compiled_graph, inputs):
    """Stream the graph and return node names plus the final accumulated state."""
    visited = []
    final_state = None
    for mode, chunk in compiled_graph.stream(
        inputs, stream_mode=["updates", "values"]
    ):
        if mode == "updates":
            visited.extend(chunk.keys())
        elif mode == "values":
            final_state = chunk
    return visited, final_state


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
    if "llm_calls" in case_input:
        inputs["llm_calls"] = case_input["llm_calls"]
    return inputs


@pytest.fixture(scope="session")
def nvidia_api_key():
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        pytest.skip("NVIDIA_API_KEY not set; skipping live LLM integration tests")
    return api_key


@pytest.fixture
def compiled_graph(nvidia_api_key):
    return create_agent_builder().compile()
