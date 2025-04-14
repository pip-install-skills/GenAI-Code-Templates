import base64

from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatBedrock(
    model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    region="us-east-1"
)

image_path = "file.jpg"
with open(image_path, 'rb') as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Please analyze the attached image and describe its content."
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_data
            }
        }
    ]
)
response = llm.invoke([message])

print(response.content)
