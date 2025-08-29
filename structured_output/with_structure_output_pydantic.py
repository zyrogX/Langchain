from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the Google Generative AI chat model
chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
)

class Review(BaseModel):
    summary: str
    sentiment: str

# Use structured output with Pydantic model
structured_model = chat_model.with_structured_output(Review)

model_response = structured_model.invoke(
    """I recently bought the new wireless earbuds, and the sound quality is fantastic.
The bass is deep, and vocals are crystal clear, making music very enjoyable.
Noise cancellation works surprisingly well in crowded places.
Battery life easily lasts a full day with moderate use.
The charging case is compact and recharges the earbuds quickly.
Bluetooth connection is stable, with no noticeable lag while streaming videos.
The touch controls are intuitive, though they sometimes misinterpret double taps.
Build quality feels premium, but the glossy finish catches fingerprints easily.
The price is slightly higher than expected, but the performance justifies it.
Overall, I’m very satisfied and would recommend it to anyone seeking quality earbuds"""
)

print(type(model_response))
print(model_response) 
