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
    inputs = build_inputs(case["input"])
    expected = case["expected"]

    visited, final_state, queries_at_search = collect_visited_nodes(compiled_graph, inputs)

    assert expected["must_reach"] in visited
    assert final_state is not None

    last_message = final_state["messages"][-1]
    assert expected["output_contains"] in str(last_message.content)
    
    if expected.get("queries_cleared", True):
        assert final_state.get("queries", []) == []
    else:
        assert len(final_state.get("queries", [])) >= expected["queries_searched_min"]
    
    if "queries_searched_min" in expected:
        assert queries_at_search is not None
        assert len(queries_at_search) >= expected["queries_searched_min"]
    
    assert final_state.get("llm_calls", 0) >= expected.get("min_llm_calls", 0)

@pytest.mark.integration
@pytest.mark.parametrize("case", NEGATIVE_CASES, ids=[c["name"] for c in NEGATIVE_CASES])
def test_incomplete_query_does_not_search(compiled_graph, case):
    """Incomplete queries should not reach search_ebay_node."""
    inputs = build_inputs(case["input"])
    expected = case["expected"]

    visited, _ , _= collect_visited_nodes(compiled_graph, inputs)

    assert expected["must_not_reach"] not in visited, (
        f"Expected not to reach {expected['must_not_reach']}, but visited: {visited}"
    )
