from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableSequence , RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def word_counter(text):
    return len(text.split())

passt = RunnablePassthrough()

prompt=PromptTemplate(
    template="Write a joke about {topic} ",
    input_variables=["topic"]
)


model = ChatMistralAI(model="mistral-medium")

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_counter) })

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result=final_chain.invoke({"topic": "AI"})    

print(result)