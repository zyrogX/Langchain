from typing import Literal, Optional
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Initialize Mistral AI model   
chat_model = ChatMistralAI(
    model="mistral-medium",  # you can also use "mistral-small" or "mistral-medium"
)

# Define Pydantic model for structured output
class ReviewModel(BaseModel):
    key_themes: list[str] = Field(..., description="List of key themes discussed in the review")
    summary : str = Field(..., description="Brief summary of the product review")
    sentiment: Literal['Good','Bad','Neutral'] = Field(..., description="Overall sentiment of the review")    
    pros: Optional[list[str]] = Field(None, description="List of pros mentioned in the review")
    cons: Optional[list[str]] = Field(None, description="List of cons mentioned in the review")
    

# Use structured output with Pydantic model
structured_model = chat_model.with_structured_output(ReviewModel)
model_response = structured_model.invoke(
    """I recently bought the new wireless earbuds, and the sound quality is fantastic.
The bass is deep, and vocals are crystal clear, making music very enjoyable.
Noise cancellation works surprisingly well in crowded places.
Battery life easily lasts a full day with moderate use.
The charging case is compact and recharges the earbuds quickly.
Bluetooth connection is stable, with no noticeable lag while streaming videos.""")

print(type(model_response))
print(model_response)