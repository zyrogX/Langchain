from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# Initialize Mistral AI model
chat_model = ChatMistralAI(
    model="mistral-small",  # you can also use "mistral-7b-instruct"
)

prompt= PromptTemplate(
    input_variables=["topic"],
    template="""Write a detailed analysis of the following Topic: {topic}""")

parser= StrOutputParser()

chain= prompt | chat_model | parser
model_response=chain.invoke(
    {"topic":"Artificial Intelligence"}
)

print(model_response)

chain.get_graph().print_ascii()