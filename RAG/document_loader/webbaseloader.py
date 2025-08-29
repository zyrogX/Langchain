from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatMistralAI(model="mistral-medium")

prompt = PromptTemplate(
    template="Answer the Following question \n {question} \n based on the following context : {data}",
    input_variables=["question","data"]
)

parser = StrOutputParser()  

url = "https://zyrogx.github.io/"

loader = WebBaseLoader(url)
docs = loader.load()

chain = prompt | model | StrOutputParser()

response=chain.invoke({"question": "introduce Iftikhar", "data": docs[0].page_content})

print(response)