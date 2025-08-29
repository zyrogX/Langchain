import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# point base_url to Cerebras endpoint
llm = ChatOpenAI(
   model="gpt-oss-120b",

    base_url="https://api.cerebras.ai/v1",   # Cerebras endpoint
)

resp = llm.invoke([HumanMessage(content="which model are you.")])
print(resp.content)
