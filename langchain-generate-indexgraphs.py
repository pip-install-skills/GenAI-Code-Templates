from langchain.indexes import GraphIndexCreator
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import networkx as nx
import io
import base64

load_dotenv()

def get_llm():
    return AzureChatOpenAI(
        openai_api_version="2023-12-01-preview",
        azure_deployment="ailm",
        max_tokens=5000,
        temperature=0.5
    )

# Sample data
employee_details = {
    "_id": "6613b1397af104338e884a9d",
    "name": "Nikhil Karewar",
    "experience": 7.4,
    "domain": "full stack development",
    "skills": [
      "core java",
      "jsp",
      "servlet",
      "spring mvc",
      "spring boot",
      "hibernate",
      "restful services",
      "angular 10",
      "sql",
      "javascript",
      "html",
      "css",
      "jquery",
      "bootstrap",
      "ajax",
      "jasper report",
      "mysql",
      "postgresql",
      "apache tomcat",
      "maven",
      "ant",
      "safbuilder",
      "github",
      "jira",
      "eclipse",
      "sts",
      "vs code",
      "windows",
      "linux"
    ]
}

jd_details = {
    "_id": "6613b1397af104338e884a9c",
    "uid": 224354,
    "job_requisition_id": 123456,
    "job_title": "Software Engineer",
    "job_role": "Full-stack Developer",
    "job_description": "We are looking for a skilled Software Engineer to join our team and help develop cutting-edge software applications. The ideal candidate will have strong programming skills and experience in both front-end and back-end development.",
    "grade": "Senior",
    "job_location": "New York, NY",
    "domain": "Information Technology",
    "skills": [
        "JavaScript",
        "React",
        "Node.js",
        "MongoDB",
        "HTML",
        "CSS",
        "Python"
    ],
    "created_at": "2024-04-08T08:56:25.700Z"
}


def generate_graph(jd_details, employee_details):
    index_creator = GraphIndexCreator(llm=get_llm())
    data = f"jd: {jd_details}\nemployee: {employee_details}"
        
    graph = index_creator.from_text(data)

    G = graph._graph
    plt.figure(figsize=(12, 8))  # Set the figure size (optional)
    nx.draw(G, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_weight='bold')

    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the image to base64
    base64_image = base64.b64encode(buffer.getvalue()).decode()

    return base64_image

print("Base64 encoded PNG image:")
print(generate_graph(jd_details, employee_details))
