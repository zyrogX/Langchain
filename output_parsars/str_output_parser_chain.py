
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",  # supports chat
    task="text-generation",                  # still works
)

model = ChatHuggingFace(llm=llm)

#detail report prompt
detail=PromptTemplate(
    input_variables=["topic"],
    template="""Write a detailed analysis of the following Topic: {topic}""")


#summary prompt
summary=PromptTemplate(
    input_variables=["text"],
    template="""Summarize the following text: {text} into 5 lines""")

parser= StrOutputParser()
chain = detail | model | parser | summary | model | parser

chain_response=chain.invoke({"topic":"Startup"})
print(chain_response)