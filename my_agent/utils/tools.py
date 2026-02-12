from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from .model import model
from .state import MainState


@tool
def scanEbay(queries: list[str]) -> str:
    """Scan eBay for items matching the query.

    Args:
        queries: The search queries.
    """
    # Implement eBay scanning logic here
    return f"Scanned eBay for '{queries}'"


REFINE_PROMPT = (
    "You are responsible an agent responsible for fetching ebay listings for pokemon cards. Please change the query to more accurately follow the format:"
    "Card quality (NM, LP, HP, ETC) or Grading Company Abbreviation(PSA, CGC, BGS, ETC)+Number Grade Set Name Card Name Card Number Release Year"
    "if there is any absent information, don't include it"
    "Here is the initial query:"
    "\n ------- \n"
    "{query}"
    "\n ------- \n"
    "please return a series of optimised query or queries, if you are returning multiple queries please separate them with a single comma, and if information about price requirements is given please put them at the front of the query with (min_price:PRICE, max_price:PRICE), if only 1 is given only put the price given"
)


@tool
def refineQuery(state: MainState):
    """Refine a search query to be more effective for eBay searching."""
    query = state["messages"][-1].content
    prompt = REFINE_PROMPT.format(query=query)
    response = model.invoke([{"role": "user", "content": prompt}])

    # Parse the response into a list of queries
    refined_queries = [q.strip() for q in response.content.split(",")]

    return {
        "filtered_queries": refined_queries,
        "messages": [HumanMessage(content=f"Refined queries: {refined_queries}")],
    }


# Export tools list
tools = [scanEbay, refineQuery]
