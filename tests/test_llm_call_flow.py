from pathlib import Path

import pytest
import yaml

from conftest import build_inputs, collect_visited_nodes

CASES_PATH = Path(__file__).parent / "cases" / "llm_call_cases.yaml"
ALL_CASES = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))

SUCCESS_CASES = [c for c in ALL_CASES if "must_reach" in c.get("expected", {})]
NEGATIVE_CASES = [c for c in ALL_CASES if "must_not_reach" in c.get("expected", {})]


@pytest.mark.integration
@pytest.mark.parametrize("case", SUCCESS_CASES, ids=[c["name"] for c in SUCCESS_CASES])
def test_llm_call_reaches_search_ebay(compiled_graph, case):
    """Complete queries should flow llm_call -> tool_node -> search_ebay_node."""
    inputs = build_inputs(case["input"])
    expected = case["expected"]

    visited, final_state = collect_visited_nodes(compiled_graph, inputs)

    assert expected["must_reach"] in visited, (
        f"Expected to reach {expected['must_reach']}, but visited: {visited}"
    )
    assert final_state is not None

    last_message = final_state["messages"][-1]
    assert expected["output_contains"] in str(last_message.content)
    assert final_state.get("filtered_queries", []) == expected["filtered_queries"]
    assert final_state.get("llm_calls", 0) >= expected["min_llm_calls"]


@pytest.mark.integration
@pytest.mark.parametrize("case", NEGATIVE_CASES, ids=[c["name"] for c in NEGATIVE_CASES])
def test_incomplete_query_does_not_search(compiled_graph, case):
    """Incomplete queries should not reach search_ebay_node."""
    inputs = build_inputs(case["input"])
    expected = case["expected"]

    visited, _ = collect_visited_nodes(compiled_graph, inputs)

    assert expected["must_not_reach"] not in visited, (
        f"Expected not to reach {expected['must_not_reach']}, but visited: {visited}"
    )
