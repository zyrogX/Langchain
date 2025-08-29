from typing import Literal
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field

# Load API keys from .env
load_dotenv()

# -------------------------
# Initialize models
# -------------------------
mistral_model = ChatMistralAI(model="mistral-medium")

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        ..., description="The sentiment of the feedback")

parser = PydanticOutputParser(pydantic_object=Feedback)
strParser=StrOutputParser()

prompt = PromptTemplate(
    template="Classify the sentiment of the following feedback as either 'positive' or 'negative': {feedback}\n{format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

feedback_chain = prompt | mistral_model | parser

prompt2 = PromptTemplate(
    template="Write an appropriate response to the following Positive feedback: {feedback} in 3 lines",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="""You are a customer support assistant.\n"
        "The customer gave negative feedback: {feedback}\n"
        "Write ONLY the reply in 3 short sentences. Do not explain your reasoning.\n"
        "Do not include commentary or meta text.""",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | mistral_model | strParser),
    (lambda x: x.sentiment == "negative", prompt3 | mistral_model | strParser),
    RunnableLambda(lambda x: "Could not determine sentiment.")  # default
)

conditional_chain = feedback_chain | branch_chain

response1 = conditional_chain.invoke({"feedback": "The product quality is bad and I am not very satisfied!"})

print(response1)
