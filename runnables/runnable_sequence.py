from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt=PromptTemplate(
    template="Write a joke about {topic} ",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the Following joke {joke}",
    input_variables=["joke"]
)


model = ChatMistralAI(model="mistral-medium")

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)

print(chain.invoke({"topic": "chickens"})) 