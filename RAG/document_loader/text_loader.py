from langchain_community.document_loaders import TextLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatMistralAI(model="mistral-medium")

prompt = PromptTemplate(
    template="Write a 4 line Summary on {data}",
    input_variables=["data"]
)

# Use raw string for Windows path
loader = TextLoader(r"RAG\mydata.txt")
docs = loader.load()

chain = prompt | model | StrOutputParser()

response = chain.invoke({"data": docs[0].page_content})

print(response)

