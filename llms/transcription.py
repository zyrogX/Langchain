import os
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
import base64
from dotenv import load_dotenv

load_dotenv()


# Initialize the model (speech-to-text)
model = ChatMistralAI(
  
    model="voxtral-mini-transcribe-2507",  # transcription model
)

# Path to your audio file
audio_file_path = r"llms\SampleInboundCall2.mp3"  # use raw string or double backslashes on Windows

# Read and base64 encode the audio
with open(audio_file_path, "rb") as f:
    audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

# Send Base64 encoded audio
response = model.invoke([
    HumanMessage(content=[
        {
            "type": "input_audio",
            "audio": audio_b64,  # base64 string, not bytes
            "format": "mp3"      # tell API what format you used
        }
    ])
])

print("Transcription:")
print(response.content)