from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

loader = PyPDFLoader("mydocument.pdf")
documents = loader.load()
print("Pages:", len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print("Chunks:", len(chunks))

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
print("Vector store created!")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context provided below.

Context:
{context}

Question:
{question}

If the answer cannot be found in the context, say:
"I don't know based on the provided document."

Do not make up information.
""")

parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | parser # structure 
)

print("\n" + "=" * 40)
print("       DOCUMENT RAG CHATBOT")
print("=" * 40)
print("Ask questions about the PDF.")
print("Type 'exit' to stop.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    answer = rag_chain.invoke(question)
    print("\nAI:", answer)
    print()