from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableSequence , RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def word_counter(text):
    return len(text.split())

passt = RunnablePassthrough()

prompt=PromptTemplate(
    template="Write a Detail note about {topic} ",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Summarize this topic in 3 lines :  {topic} ",
    input_variables=["topic"]
)


model = ChatMistralAI(model="mistral-medium")

parser = StrOutputParser()

report = RunnableSequence(prompt, model, parser)



branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 50, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report, branch_chain)
result=final_chain.invoke({"topic": "India vs Pakistan"})
print(result)