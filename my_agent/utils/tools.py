from langchain_core.tools import tool


@tool
def scanEbay(query: str) -> str:
    """Scan eBay for items matching the query.
    
    Args:
        query: The search query.
    """
    # Implement eBay scanning logic here
    return f"Scanned eBay for '{query}'"
    



# Export tools list
tools = [scanEbay]
