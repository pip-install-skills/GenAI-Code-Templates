# Optimized sample code to use the Azure OpenAI Service's Whisper API from Python
# LICENSE: MIT License

import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set parameters for Azure OpenAI Service Whisper
openai.api_type = os.getenv("OPENAI_API_TYPE", "azure")
openai.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_version = os.getenv("OPENAI_API_VERSION")
deployment_id = os.getenv("AZURE_DEPLOYMENT_ID")

# Music files are copyrighted by MaouDamashii
# Redistribution and modification are permitted under the following rules page.
# https://maou.audio/rule/
audio_file_path = r".\path\to\audio\file"

# Read and transcribe the audio file
with open(audio_file_path, "rb") as audio_file:
    transcript = openai.audio.transcriptions.create(
        model=deployment_id,
        file=audio_file,
        language="el"
    )
# Print transcription results
print(transcript.text)
