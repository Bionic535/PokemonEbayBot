import operator
from typing import Annotated, List

from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict


class MessagesState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    llm_calls: int
