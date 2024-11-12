# # %%
# !pip install langchain langchain_community duckduckgo-search google-serp-api wikipedia

# # %%
# !pip install --upgrade langchain

# # %%
# !pip install -U langchain-openai

# %%
import pandas as pd
import os
import json
import openai
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool
from langchain_community.agent_toolkits import JsonToolkit, create_json_agent
from langchain_community.tools.json.tool import JsonSpec
import logging
from multiprocessing import Pool, Manager, Lock
from functools import partial
import time
from tqdm import tqdm
import csv
import sys
from langchain.agents import AgentExecutor

# Initialize logging to output to a file rather than the terminal
log_file_path = './log/agent_process.log'  # Log file path for all logs

# Configure logging to include the desired log level and output to a file
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs if needed
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),  # Log output to file
        logging.StreamHandler()  # Optionally, keep logging to console if needed
    ]
)

# If you want to redirect standard output (print statements) to the same log file:
sys.stdout = open(log_file_path, 'a')  # Redirect stdout to the log file
sys.stderr = open(log_file_path, 'a')  # Redirect stderr to the log file for error messages

# %%
def setup_agent(openai_api_key):
    """Initialize and return a LangChain agent with DuckDuckGo search capabilities."""
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
    duckduckgo_search = DuckDuckGoSearchRun()
    
    tools = [
        Tool(
            name="DuckDuckGo Search",
            func=duckduckgo_search.run,
            description="Useful for retrieving detailed information based on a given topic."
        )
    ]
    
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True
    )
    
# %%

def setup_json_agent(openai_api_key, json_data):
    """Initialize and return a LangChain agent with JSON interaction capabilities."""
    
    # Initialize the language model (LLM)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
    
    # Create a JSON specification from the provided JSON data
    json_spec = JsonSpec(dict_=json_data, max_value_length=4000)
    
    # Create a toolkit for interacting with the JSON data
    json_toolkit = JsonToolkit(spec=json_spec)
    
    # Create the JSON agent using the LLM and toolkit
    json_agent_executor = create_json_agent(
        llm=llm,
        toolkit=json_toolkit,
        verbose=True
    )
    
    return json_agent_executor

# %%
def search_and_summarize(agent, topic):
    """Perform detailed search and generate summary."""
    try:
        prompt = (
            f"Search for detailed and technical information on the topic: '{topic}' "
            "from a medical standpoint. Provide a comprehensive, detailed summary in paragraph form. "
            "Avoid using bullet points or lists. The summary should focus on technical and medical aspects "
            "with a high level of detail, suitable for a medical professional audience."
        )
        summary = agent.run(prompt)
        return summary
    except Exception as e:
        logging.error(f"Error summarizing topic '{topic}': {e}")
        return None

# %%
def generate_claim_triplet(agent, summary, num_samples=3):
    """Generate claim triplets from a summary."""
    try:
        prompt = """
        Using the following detailed summary, generate three types of claims:\n\nSummary:\n{summary}\n\n1) A supporting claim that paraphrases a key assertion.\n2) A contradictory claim that directly contradicts a key evidence provided in the summary.\n3) An ambiguous claim that either partially supports or contradicts, or presents elements that are neither clearly supported nor contradicted.\n\nEach claim should be one or two sentences long. Ideally, the claims should be generated from different key assertions or sections of the summary.\n\nReturn ONLY the claims in this exact JSON format below. DO NOT include any extra text or explanations. DO NOT add ```json``` formatting. Just output the exact JSON as a string.\n[{{\n  \"supporting_claim\": '...',\n  \"contradictory_claim\": '...',\n  \"ambiguous_claim\": '...'\n}}].\n\n Output {num_samples} triplets of supporting, contradictory, and ambiguous claims from the provided summary. Give all {num_samples} triplets in the same JSON array.\n\n\n\n

        Summary:
        {summary}
        """
        # response = agent.run(prompt)
        llm = OpenAI(openai_api_key=openai_api_key)
        prompt = PromptTemplate(
            input_variables=["summary", "additional_details"],
            template=prompt
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        response = llm_chain.run({"summary": summary, "num_samples": 2})
        
        print("JSON Response", response)
        
        response = json.loads(response)
        
        return response
    
    except Exception as e:
        logging.error(f"Error generating claims for summary: {e}")
        return "N/A", "N/A", "N/A"


# %%
def write_row(file_path, row_dict, file_lock):
    """Write a single row to the CSV file in a thread-safe manner."""
    with file_lock:
        file_exists = os.path.exists(file_path)
        mode = 'a' if file_exists else 'w'
        
        with open(file_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)

# %%
def process_topic(args):
    """Process a single topic and write results immediately."""
    topic, openai_api_key, processed_topics, file_lock, output_file, log_file = args
    
    print("Processing:", topic)
    
    if topic in processed_topics:
        return None
    
    try:
        agent = setup_agent(openai_api_key)
        summary = search_and_summarize(agent, topic)
        
        if summary:
            # supports, contradicts, ambiguous = generate_claim_triplet(agent, summary)
            
            responses = generate_claim_triplet(agent, summary)
            
            # print("Got responses", responses)
            
            for response in responses:
                # response = json.loads(response)
                
                # print("Writing to file", response)
                
                supports = response.get("supporting_claim", "N/A")
                contradicts = response.get("contradictory_claim", "N/A")
                ambiguous = response.get("ambiguous_claim", "N/A")
            
                # Prepare result row
                result = {
                    'Topic': topic,
                    'Evidence': summary,
                    'Supports': supports,
                    'Contradicts': contradicts,
                    'Ambiguous': ambiguous
                }
                
                # Write to output file immediately
                write_row(output_file, result, file_lock)
                
                # Write to log file
                write_row(log_file, {'Processed_Topic': topic}, file_lock)
            
    except Exception as e:
        logging.error(f"Error processing topic '{topic}': {e}")
    
    return None

# %%
def init_files(output_file, log_file):
    """Initialize output and log files with headers if they don't exist."""
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Topic', 'Evidence', 'Supports', 'Contradicts', 'Ambiguous'])
            writer.writeheader()
    
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Processed_Topic'])
            writer.writeheader()

# %%
# File paths
log_file = './csv/process_log.csv'
output_file = './csv/generated_claim_triplets_with_topics.csv'
input_file = "./csv/categorized_content_links_unique.csv"

# OpenAI API key
# openai_api_key = "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"
# openai_api_key = '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'
# openai_api_key = "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"
openai_api_key = os.environ['OPENAI_API_KEY']

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize files
init_files(output_file, log_file)

# Load data
df = pd.read_csv(input_file)
topics = df.iloc[:, 1].astype(str) + " " + df.iloc[:, 2].astype(str)

# %%
print(topics)

# %%
if __name__ == "__main__":
    # Load processed topics
    processed_topics = set()
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        processed_topics = set(log_df["Processed_Topic"])
    
    # Initialize multiprocessing resources
    manager = Manager()
    file_lock = manager.Lock()
    num_processes = min(os.cpu_count() - 1, 4)  # Use up to 4 processes or CPU count - 1
    
    # Prepare arguments for each topic
    process_args = [
        (topic, openai_api_key, processed_topics, file_lock, output_file, log_file)
        for topic in topics
    ]
    
    # Process topics in parallel with immediate output
    with Pool(num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(process_topic, process_args), total=len(topics), file=sys.__stderr__):
            pass
