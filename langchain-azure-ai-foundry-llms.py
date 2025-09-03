
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate

llm = AzureAIChatCompletionsModel(
    model_name="grok-3",
    endpoint="https://[your-service].services.ai.azure.com/openai/deployments/grok-3",
    credential="your-api-key",
    api_version="2025-01-01-preview",
)

msgs = [
    SystemMessage("You’re a witty assistant."),
    HumanMessage("Give me a one-liner about Mondays with an emoji.")
]

resp = llm.invoke(msgs)
print(resp.content)

print("---")

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Give me a one-liner about {topic} with an emoji."
)

chain = prompt | llm

response = chain.invoke({"topic": "Mondays"})
print(response)

"""
Output:

Mondays are the ultimate test of willpower—just survive! 😅
---
content='Mondays are a fresh start to conquer the week! 🌞' additional_kwargs={} response_metadata={'model': 'grok-3', 'token_usage': {'input_tokens': 18, 'output_tokens': 13, 'total_tokens': 31}, 'finish_reason': 'stop'} id='run--626cb0fa-b9d2-4501-b948-bb591f2e66fd-0' usage_metadata={'input_tokens': 18, 'output_tokens': 13, 'total_tokens': 31}
"""
