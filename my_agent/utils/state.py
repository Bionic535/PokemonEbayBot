import operator

from langchain.messages import AnyMessage
from typing_extensions import Annotated, TypedDict
class QuerySpec(TypedDict, total=False):
    Pkmn_Name: str
    Type: str
    Grade Company: str
    Grade: int
    Min Price: int
    Max Price: int
    Set: str
    
    
    
def merge_query(left: QuerySpec | None, right: QuerySpec | None) -> QuerySpec:
    if left is None:
        return right
    if right is None:
        return left
    return {**left, **right}

class MainState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    query: Annotated[QuerySpec, merge_query]
    items: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    queries: Annotated[list[QuerySpec], operator.add]