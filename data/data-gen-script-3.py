import os
import json
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.tools import DuckDuckGoSearchResults

# Step 1: Define the search tool using DuckDuckGo
class DuckDuckGoSearchTool:
    def __init__(self):
        self.search_tool = DuckDuckGoSearchResults()

    def search(self, query):
        # Perform the DuckDuckGo search
        search_results = self.search_tool.run(query)
        return search_results

# Step 2: Define the OpenAI processor with two prompts (for summarizing and JSON formatting)
class OpenAIProcessor:
    def __init__(self, openai_api_key):
        self.llm = OpenAI(openai_api_key=openai_api_key)

    # Step 2.1: First prompt - Summarizing the search results
    def summarize(self, search_results, summary_prompt_template):
        prompt = PromptTemplate(
            input_variables=["search_results"],
            template=summary_prompt_template
        )
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        summary = llm_chain.run({"search_results": search_results})
        return summary

    # Step 2.2: Second prompt - Processing the summary into JSON
    def generate_json(self, summary, json_prompt_template, additional_details):
        prompt = PromptTemplate(
            input_variables=["summary", "additional_details"],
            template=json_prompt_template
        )
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        result_json = llm_chain.run({"summary": summary, "additional_details": additional_details})
        return result_json

# Step 3: Combine search, summarize, and generate JSON in a Langchain workflow
def search_summarize_generate_json(query, openai_api_key, summary_prompt_template, json_prompt_template, additional_details):
    # Initialize DuckDuckGo search tool and OpenAI processor
    search_tool = DuckDuckGoSearchTool()
    processor = OpenAIProcessor(openai_api_key=openai_api_key)

    # Step 3.1: Search using DuckDuckGo
    search_results = search_tool.search(query)

    # Step 3.2: Summarize the search results using the first OpenAI prompt
    summary = processor.summarize(search_results, summary_prompt_template)
    
    print(summary)

    # Step 3.3: Generate the JSON from the summary using the second OpenAI prompt
    result_json = processor.generate_json(summary, json_prompt_template, additional_details)

    return result_json

# Example usage
if __name__ == "__main__":
    
    log_file = './csv/process_log.csv'
    output_file = './csv/generated_claim_triplets_with_topics.csv'
    input_file = "./csv/categorized_content_links_unique.csv"
    
    df = pd.read_csv(input_file)
    topics = df.iloc[:, 1].astype(str) + " " + df.iloc[:, 2].astype(str)
    
    openai_api_key = os.environ['OPENAI_API_KEY']
    query = "Anaphylaxis Food-induced anaphylaxis medical overview pathophysiology diagnosis management"

    # Define the prompt for summarizing search results
    summary_prompt_template = """
    You are an expert assistant. Based on the following search results, summarize the main findings clearly and concisely.

    Search Results:
    {search_results}

    Provide a detailed summary:
    """

    # Define the second prompt for generating JSON based on the summary
    json_prompt_template = """
    Using the following detailed summary, generate three types of claims:\n\nSummary:\n{summary}\n\n1) A supporting claim that paraphrases a key assertion.\n2) A contradictory claim that directly contradicts a key evidence provided in the summary.\n3) An ambiguous claim that either partially supports or contradicts, or presents elements that are neither clearly supported nor contradicted.\n\nEach claim should be one or two sentences long. Ideally, the claims should be generated from different key assertions or sections of the summary.\n\nReturn ONLY the claims in this exact JSON format below. DO NOT include any extra text or explanations. DO NOT add ```json``` formatting. Just output the exact JSON as a string.\n[{{\n  'supporting_claim': '...',\n  'contradictory_claim': '...',\n  'ambiguous_claim': '...'\n}}].\n\n Output 3 triplets of supporting, contradictory, and ambiguous claims from the provided summary. Give all 3 triplets in the same JSON array.\n\n\n\n

    Summary:
    {summary}
    """

    # Define any additional details to include in the JSON output
    additional_details = {
        "source": "DuckDuckGo",
        "confidence_score": 0.95,
        # "related_topics": ["AI Agents", "OpenAI", "Web Search"]
    }

    # Run the full process and output the final JSON
    result_json = search_summarize_generate_json(query, openai_api_key, summary_prompt_template, json_prompt_template, additional_details)
    print(result_json)
