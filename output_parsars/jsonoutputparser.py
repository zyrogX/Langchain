
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # supports chat
    task="text-generation",                  # still works
)

model = ChatHuggingFace(llm=llm)
parser= JsonOutputParser()
template1= PromptTemplate(
    input_variables=[],
    template="give me a name , age and city of fictional person /n {format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions(),
                       })


chain= template1 | model | parser
result=chain.invoke({})

print(result)