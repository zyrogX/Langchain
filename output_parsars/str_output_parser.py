
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # supports chat
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

prompt1=detail.format(topic="Artificial Intelligence")

model_response1=model.invoke(prompt1)

summary_prompt=summary.format(text=model_response1)

summary_response=model.invoke(summary_prompt)

print(summary_response)