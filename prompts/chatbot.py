from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

model= ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",

)

chat_history = [SystemMessage(content="You are a helpful assistant.")]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        print("Exiting the chat. Goodbye!")
        break   
    model_response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=model_response.content))
    print(f"Model: {model_response.content}")
    
print("Chat history:" + str(chat_history))  # Print chat history for reference