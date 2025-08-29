from langchain.text_splitter import RecursiveCharacterTextSplitter

text = '''LangChain is a framework for developing applications powered by language models. It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more.
It is designed to help you build applications that are:
- Data-aware: connect a language model to other sources of data
- Agentic: allow a language model to interact with its environment
- Modular: combine components in a flexible way to create custom applications
The core idea of the library is that we can "chain" together different components to create more advanced use cases around LLMs. Chains may consist of multiple components from several modules: prompts, models, indexes, memory, and more.
The library is modularized to allow for easy use of these components, as well as easy integration of external tools and APIs. The end goal is to make it as easy as possible to build and deploy complex LLM-powered applications.      

'''

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=0
)

chunks = splitter.split_text(text)  
print(len(chunks))
print(chunks)

