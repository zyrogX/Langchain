from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} Expert.'),
    ('human', 'Explain what is {question}'),
   
    
])

prompt=chat_template.invoke({'domain': "AI", 'question': "RLHF"})

print(prompt)