from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableSequence , RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

passt = RunnablePassthrough()

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

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2,model, parser)})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result=final_chain.invoke({"topic": "AI"})    

print(result)