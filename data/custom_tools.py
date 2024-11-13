from duckduckgo_search import DDGS
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from typing import Optional, List

# Custom DuckDuckGo search tool
class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = (
        "A wrapper around DuckDuckGo Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )

    def _run(self, query: str) -> str:
        """Execute the DuckDuckGo search and return results as a string."""
        proxy = "http://ratligqioa327538:Ng7Iy8KV6vTuRrN5@isp2.hydraproxy.com:9989"
        ddg = DDGS(proxy=proxy)
        results = ddg.text(query, max_results=5)
        if not results:
            return "No results found."
        return "".join([f"Title: {res['title']}. {res['body']}" for res in results])

    async def _arun(self, query: str) -> str:
        """Run the search asynchronously."""
        raise NotImplementedError("Async search is not supported yet.")

# Example prompt template to use with the search tool
# prompt_template = PromptTemplate(
#     input_variables=["topic"],
#     template="Use DuckDuckGo to search for the following topic: {topic}."
# )

# # Example of integrating the search tool into a LangChain agent
# def use_duckduckgo_search_tool(query: str):
#     # Create an instance of the search tool
#     search_tool = DuckDuckGoSearchTool()
#     return search_tool.run(query)

# # Example of how to use it
# if __name__ == "__main__":
#     topic = "Latest advancements in AI for healthcare"
#     search_results = use_duckduckgo_search_tool(topic)
#     print(search_results)
