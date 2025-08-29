from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv  
import streamlit as st
from langchain_core.prompts import PromptTemplate


load_dotenv()   

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",   
)


st.header("Reasearch Assistant")


research_paper = st.selectbox(
    "Choose a research paper:",
    [
        "Attention Is All You Need (Vaswani et al., 2017)",
        "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
        "AlphaGo (Silver et al., 2016)",
        "GANs: Generative Adversarial Nets (Goodfellow et al., 2014)",
        "YOLO: You Only Look Once (Redmon et al., 2016)",
        "ImageNet Classification with Deep Convolutional Neural Networks (Krizhevsky et al., 2012)",
        "RLHF: Training Language Models to Follow Instructions (Ouyang et al., 2022)",
        "Segment Anything Model (Kirillov et al., 2023)",
    ]
)


# Select Tone
tone = st.selectbox(
    "Choose tone:",
    ["Professional and academic", "Casual", "Neutral", "Critical", "Simplified"]
)

# Select Length
length = st.selectbox(
    "Choose summary length:",
    ["Short (100 words)", "Medium (200 words)", "Detailed (500+ words)", "Bullet points"]
)

# Select Style
style = st.selectbox(
    "Choose summary style:",
    ["Concise overview", "In-depth explanation", "Key insights only", "Layman-friendly"]
)


template= PromptTemplate(
    input_variables=["research_paper", "tone", "length", "style"],
    template="""
    You are a research assistant. Summarize the following research paper in a {length} format with a {tone} tone and {style} style.
    
    Research Paper: {research_paper}
    """,
)

prompt=template.invoke(
    {
        "research_paper": research_paper,
        "tone": tone,
        "length": length,
        "style": style
    }
)

if st.button("Summarize"):
    result = model.invoke(prompt)
    st.write(result.content)

