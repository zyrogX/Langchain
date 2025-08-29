from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

# Initialize model
llm = ChatMistralAI(
    model="mistral-medium",  # models: mistral-tiny, mistral-small, mistral-medium
 
)

# Define a prompt
prompt = ChatPromptTemplate.from_template("Explain LangChain in simple terms.")

# Run the model
response = llm.invoke(prompt.format_messages())
print(response.content)
