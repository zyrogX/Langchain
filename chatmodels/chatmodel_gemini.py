from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

model= ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",

)


result=model.invoke(
    "What is the capital of Pakistan?"

)

print(result.content)