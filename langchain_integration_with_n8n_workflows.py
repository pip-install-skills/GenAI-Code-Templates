import os
import uuid
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain import hub

from app.classes.base_llm_class import BaseLLMClass

# Load environment variables
load_dotenv(override=True)

# Constants for environment variables
WEBHOOK_USERNAME = os.getenv("WEBHOOK_USERNAME")
WEBHOOK_PASSWORD = os.getenv("WEBHOOK_PASSWORD")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "http://localhost:5678/webhook/1ce93035-8e44-4b02-84f8-42944990c59e")

# Initialize base LLM class
base_llm_class = BaseLLMClass()

@tool
def add(a: int, b: int) -> int:
    """
    This tool adds two integers and returns the sum.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The sum of the two numbers.
    """
    return a + b

@tool
def websearch(question: str, session_id: str = None) -> dict:
    """
    This tool performs an internet search and returns the response.

    Args:
        question (str): The query to search for.
        session_id (str, optional): A unique session identifier. Defaults to None, which generates a new UUID.

    Returns:
        dict: The search result or error details.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    payload = {
        "chatInput": question,
        "sessionId": session_id
    }

    try:
        # Make a POST request to the webhook
        response = requests.post(
            WEBHOOK_URL, 
            json=payload, 
            auth=HTTPBasicAuth(WEBHOOK_USERNAME, WEBHOOK_PASSWORD)
        )

        # Handle response
        if response.status_code == 200:
            return response.json().get("output", {})
        else:
            return {"error": f"Received status code {response.status_code}", "details": response.text}

    except requests.RequestException as e:
        # Catch and report request errors
        return {"error": "Request failed", "details": str(e)}

def main():
    query = input("Enter question to search the internet: ")

    # List of tools to use in the agent
    tools = [websearch, add]

    # Initialize the LLM
    llm = base_llm_class.get_llm()

    # Pull the prompt template from the hub
    prompt_template = hub.pull("hwchase17/react")
    print(prompt_template)

    # Create the agent with the tools and LLM
    agent = create_react_agent(llm, tools, prompt_template)

    # Execute the agent with the query input
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    response = agent_executor.invoke({"input": query})

    # Output the response
    print(response)

if __name__ == "__main__":
    main()
