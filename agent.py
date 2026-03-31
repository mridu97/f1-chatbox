from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

print("Loading F1 knowledge base...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are an expert F1 assistant helping fans understand Formula 1.
Use the following context from the F1 regulations to answer the question.
If you don't know the answer from the context, say so honestly.

Context:
{context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n🏎️  F1 Agent ready! Type your question or 'quit' to exit.\n")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    answer = chain.invoke(question)
    print(f"\nF1 Agent: {answer}\n") 