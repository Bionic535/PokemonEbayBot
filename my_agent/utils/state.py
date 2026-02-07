from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class MainState(TypedDict):
    searchquery: str
    items: Annotated[list[AnyMessage], operator.attrgetter('content')]
    