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

image_path = r"image.jpg"

base64_image = encode_image(image_path)

prompt_instruction = """
You are an OCR assistant expert in both Greek and English languages. Your task is to read the content provided from OCR text, extract relevant information, and present it in a structured format. Follow these instructions carefully:

1. You will be given OCR text that may contain a mix of Greek and English language

2. Carefully read and analyze the provided text.

3. Organize the extracted information into a structured format.

4. Present your final output in the following order:
   a. Extracted information (in the structured format described above)
   b. Summary

Remember to maintain the original spelling and formatting of the extracted text as much as possible. If the OCR text contains any obvious errors or inconsistencies, make a note of them but do not attempt to correct them unless you are absolutely certain of the correct form.

Begin your analysis and extraction now, and provide your output below:
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
            #{"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64_image}"}
            }
        ]}
    ],
    temperature=0.7,
)
print(response.choices[0].message.content)
