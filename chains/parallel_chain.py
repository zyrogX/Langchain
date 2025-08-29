from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Load API keys from .env
load_dotenv()

# -------------------------
# Initialize models
# -------------------------
mistral_model = ChatMistralAI(
    model="mistral-tiny",        # good for detailed notes
            # increase timeout
)

gemini_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",      # fast for quizzes
    temperature=0.7
)

# -------------------------
# Define prompts
# -------------------------
notes_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write detailed study notes on the following topic: {topic}"
)

quiz_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate a 5-question multiple choice quiz on the following topic: {topic}"
)

merge_prompt = PromptTemplate(
    input_variables=["notes", "quiz"],
    template=(
        "Combine the following into a single structured document.\n\n"
        "=== Notes ===\n{notes}\n\n"
        "=== Quiz ===\n{quiz}\n"
    )
)

# -------------------------
# Output parser
# -------------------------
parser = StrOutputParser()

# -------------------------
# Build parallel chain
# -------------------------
parallel_chain = RunnableParallel(
    {
        "notes": notes_prompt | mistral_model | parser,
        "quiz": quiz_prompt | gemini_model | parser,
    }
)

# Merge the results
merge_chain = merge_prompt | gemini_model | parser

# Final pipeline: parallel → merge
chain = parallel_chain | merge_chain

# -------------------------
# Run the chain
# -------------------------
if __name__ == "__main__":
    response = chain.invoke({"topic": "Cryptocurrency"})
    print("=== Final Output ===\n")
    print(response)

    # If supported by your LangChain version
    try:
        chain.get_graph().print_ascii()
    except Exception as e:
        print("Graph visualization not available:", e)
