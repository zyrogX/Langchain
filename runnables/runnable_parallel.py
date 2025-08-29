from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableSequence , RunnableParallel
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template= "Genterate a Tweet about {topic} ",
    input_variables=["topic"]   
)

prompt2 = PromptTemplate(
    template= "Genterate a linkedin Post about {topic} ",
    input_variables=["topic"]   
)

model = ChatMistralAI(model="mistral-medium")

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin_post': RunnableSequence(prompt2, model, parser)
    
})

result=parallel_chain.invoke({"topic": "AI"})

print(result)