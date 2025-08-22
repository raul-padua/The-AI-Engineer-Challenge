from typing import List, Optional
import os

try:
    from tavily import TavilyClient
    _TAVILY_AVAILABLE = True
except Exception:
    TavilyClient = None  # type: ignore
    _TAVILY_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # If dotenv is not installed, ignore; environment variables may still be set
    pass


class TavilySearch:
    """
    Thin wrapper around Tavily web search. If the library or API key is not
    available, methods degrade gracefully and return an empty list.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.enabled = _TAVILY_AVAILABLE and bool(self.api_key)
        self._client = TavilyClient(api_key=self.api_key) if self.enabled else None

    def search_snippets(self, query: str, max_results: int = 3) -> List[str]:
        """
        Perform a web search and return a list of content snippets.

        Returns an empty list if Tavily is unavailable or not configured.
        """
        if not self.enabled or self._client is None:
            return []

        try:
            # Tavily response contains a list of results with fields like
            # {"title", "url", "content"}
            result = self._client.search(query=query, max_results=max_results)
            items = result.get("results", []) if isinstance(result, dict) else []
            snippets: List[str] = []
            for item in items:
                content = item.get("content") or ""
                title = item.get("title") or ""
                if content:
                    snippets.append(f"Title: {title}\n{content}")
            return snippets[:max_results]
        except Exception:
            # Fail softly: treat web search as optional context
            return []


