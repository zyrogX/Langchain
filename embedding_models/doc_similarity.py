from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

documents = [
    "Pakistan is in asia",
    "Islamabad is the capital of Pakistan",
    "Pakistan is a democratic country",
    "Pakistan is a nuclear power",
    "Paris is the capital of France",
    "Delhi is the capital of india"
]


question = "What is the capital of france?"

# Embed the documents and the question
doc_embeddings = embedding.embed_documents(documents)
question_embedding = embedding.embed_query(question)

# Calculate cosine similarity between the question and each document
similarities = cosine_similarity([question_embedding], doc_embeddings)
# Get the index of the most similar document
most_similar_index = similarities.argmax()      
# Print the most similar document
print(f"{question}: {documents[most_similar_index]}")