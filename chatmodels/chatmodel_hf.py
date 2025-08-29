import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # supports chat
    task="text-generation",                  # still works
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of Pakistan?")
print(result.content)
