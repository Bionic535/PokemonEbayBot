import logging

from langchain_core.tools import tool
from my_agent.utils.state import QuerySpec

from .state import MainState

logger = logging.getLogger(__name__)
REQUIRED = ["Pokemon_Name", "Type"]

def is_query_complete(q: QuerySpec) -> bool:
    if not all(q.get(k) for k in REQUIRED):
        return False
    if q.get("Type") == "graded" and not q.get("Grade_Company") and not q.get("Grade"):
        return False
    if not q.get("Min_Price") and not q.get("Max_Price"):
        return False
    return True

@tool
def canSearch():
    """Check if the current query is complete enough to search for."""
    return {"can_search": True}

@tool
def finalizeQuery(query: QuerySpec):
    """Mark the current query as complete and queue it for search."""
    if not is_query_complete(query):
        missing = [k for k in REQUIRED if not query.get(k)]
        if query.get("Type") == "graded" and not query.get("Grade_Company") and not query.get("Grade"):
            missing.append("Grade_Company")
            missing.append("Grade")
        if not query.get("Min_Price") and not query.get("Max_Price"):
            missing.append("Min_Price or Max_Price")
        return {
            "query": query,
            "complete": False,
            "missing_fields": missing,
        }
    return {"queries": [query], "query": {}, "complete": True}

@tool
def scanEbay(queries: list[QuerySpec]) -> str:
    """Scan eBay for items matching the query.

    Args:
        queries: The search queries.
    """
    # Implement eBay scanning logic here
    return f"Scanned eBay for '{queries}'"


@tool
def updateQuery(
    Pokemon_Name: str | None = None,
    Type: str | None = None,
    Grade_Company: str | None = None,
    Grade: int | None = None,
    Min_Price: int | None = None,
    Max_Price: int | None = None,
    Set_Name: str | None = None,
):
    """Update the in-progress search query with any newly known fields."""
    updates = {
        k: v
        for k, v in {
            "Pokemon_Name": Pokemon_Name,
            "Type": Type,
            "Grade_Company": Grade_Company,
            "Grade": Grade,
            "Min_Price": Min_Price,
            "Max_Price": Max_Price,
            "Set_Name": Set_Name,
        }.items()
        if v is not None
    }
    return {"query": updates}


tools = [updateQuery, finalizeQuery, canSearch]
