
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # supports chat
    task="text-generation",                  # still works
)

model = ChatHuggingFace(llm=llm)


schema=[ResponseSchema(name="fact_1", description="fact 1 about the topic"),
        ResponseSchema(name="fact_2", description="fact 2 about the topic"),
        ResponseSchema(name="fact_3", description="fact 3 about the topic"),]

parser= StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(
    input_variables=["topic"],  
    template="give me 3 facts about the following topic {topic} /n {format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()}      )
 
chain = template | model | parser
result=chain.invoke({"topic":"Artificial Intelligence"})
print(result)