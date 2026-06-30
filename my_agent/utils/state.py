import operator

from langchain.messages import AnyMessage
from typing_extensions import Annotated, TypedDict, Literal
class QuerySpec(TypedDict, total=False):
    Pokemon_Name: str
    Type: Literal["raw", "graded"]
    Grade_Company: str
    Grade: int
    Min_Price: int
    Max_Price: int
    Set_Name: str
    
    
    
def merge_query(left: QuerySpec | None, right: QuerySpec | None) -> QuerySpec:
    if right is None:
        return left or {}
    if left is None:
        return right or {}
    if not right:
        return {}
    return {**left, **right}


def merge_queries(
    left: list[QuerySpec] | 
    None, right: list[QuerySpec] | None
    ) -> list[QuerySpec]:
    if right is None:
        return left or []
    if not right:
        return []
    return (left or []) + right

class MainState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    query: Annotated[QuerySpec, merge_query]
    llm_calls: int
    queries: Annotated[list[QuerySpec], merge_queries]
    can_search: bool