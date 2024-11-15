from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

import os

load_dotenv()

os.environ["LLAMA_CLOUD_API_KEY"] = "<llamaparse-api-key>"

# set up parser
parser = LlamaParse(
    result_type="markdown"  # "markdown" and "text" are available
)

# use SimpleDirectoryReader to parse our file
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    input_files=[r"path/too/file.pdf"], 
    file_extractor=file_extractor
).load_data()

# Save the output to a file
output_file = r"path/to/output.md"  # Change path if needed
with open(output_file, "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(str(doc) + "\n")  # Convert doc to string if necessary

print(f"Output saved to {output_file}")
