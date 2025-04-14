from openai import OpenAI, AzureOpenAI
import os
from dotenv import load_dotenv
import base64

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image

image_path = r"file.jpg"

base64_image = encode_image(image_path)

prompt_instruction = """
Extract text from the file
"""

client = OpenAI()
azure_client = AzureOpenAI(
    azure_deployment="gpt-4o",
    api_version="2025-01-01-preview",
)

MODEL="gpt-4o"

response = azure_client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": prompt_instruction},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64_image}"}
            }
        ]}
    ],
    temperature=0.7,
)
print(response.choices[0].message.content)
