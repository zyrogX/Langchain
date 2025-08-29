
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # supports chat
    task="text-generation",                  # still works
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="name of the person")
    age: int = Field(gt=18,description="age of the person")
    city: str = Field(description="city of the person")
    
parser= PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
    template="give me a name , age and city of fictional {place} person /n {format_instructions}",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()})

print(template)

chain = template | model | parser
result=chain.invoke({"place":"Pakistan"})
print(result)
print(type(result))